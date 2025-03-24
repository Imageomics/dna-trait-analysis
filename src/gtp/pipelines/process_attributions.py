import dataclasses
import os

import click
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gtp.configs.loaders import load_configs
from gtp.configs.project import GenotypeToPhenotypeConfigs
from gtp.dataloading.data_collectors import load_training_data
from gtp.dataloading.datasets import GTP_Dataset
from gtp.dataloading.path_collectors import get_experiment_directory
from gtp.dataloading.tools import save_json
from gtp.evaluation import (
    AttributionMethod,
    get_lrp_attr,
    get_perturb_attr,
    get_windowed_edit_attr,
)
from gtp.models.net import SoyBeanNet
from gtp.options.process_attribution import ProcessAttributionOptions
from gtp.tools.calculation import gather_model_predictions_and_actuals


def _get_evaluation_metrics(model, dataloader):
    actual, predicted = gather_model_predictions_and_actuals(model, dataloader)
    pearson_stats = pearsonr(predicted, actual)
    diffs = np.array(actual) - np.array(predicted)
    rmse = np.sqrt(np.square(diffs).sum() / diffs.shape[0])
    return {
        "rmse": rmse,
        "pvalue": pearson_stats.pvalue,
        "pearsonr": pearson_stats.statistic,
    }


def _process_chromosome(
    configs: GenotypeToPhenotypeConfigs,
    options: ProcessAttributionOptions,
    chromosome: int,
):
    if chromosome is not None:
        options = dataclasses.replace(options)
        options.chromosome = chromosome

    experiment_dir = get_experiment_directory(
        configs.io,
        species=options.species,
        wing=options.wing,
        color=options.color,
        chromosome=options.chromosome,
        exp_name=options.exp_name,
    )

    if not os.path.exists(experiment_dir):
        print(
            f"{experiment_dir} does not exist. Unable to process chromosome {options.chromosome}"
        )
        return

    # TODO: load the saved data instead of recomputing

    train_data, val_data, test_data = load_training_data(configs, options)
    num_vcfs = train_data[0].shape[1]
    model = SoyBeanNet(
        window_size=num_vcfs,
        num_out_dims=options.out_dims,
        insize=options.insize,
        hidden_dim=options.hidden_dim,
        drop_out_prob=options.drop_out_prob,
    )

    model.load_state_dict(torch.load(experiment_dir / "model.pt", weights_only=True))
    model = model.cuda()
    model.eval()

    for data, phase_str in [
        (train_data, "training"),
        (val_data, "validation"),
        (test_data, "test"),
    ]:
        dset = GTP_Dataset(*data)
        dloader = DataLoader(
            dset, batch_size=options.batch_size, num_workers=options.num_workers
        )

        targets = [
            idx + options.out_dims_start_idx_attribution
            for idx in range(options.out_dims_attribution)
        ]

        match options.attr_method:
            case AttributionMethod.LRP.value:
                attribution_data = get_lrp_attr(
                    model,
                    dloader,
                    targets=targets,
                    verbose=options.verbose,
                    num_processes=8,
                )
            case AttributionMethod.PERTURB.value:
                attribution_data = get_perturb_attr(
                    model, dloader, targets=targets, verbose=options.verbose
                )
            case AttributionMethod.WINDOWED_EDITING.value:
                edits = [
                    torch.tensor([0, 0, 1]),  # AA
                    torch.tensor([0, 1, 0]),  # Aa/aA
                    torch.tensor([1, 0, 0]),  # aa
                    torch.tensor([0, 0, 0]),  # zero-out
                ]
                attribution_data = get_windowed_edit_attr(
                    model,
                    dloader,
                    edits=edits,
                    window=options.window_size,
                    verbose=options.verbose,
                )
            case _:
                raise NotImplementedError(
                    f"{options.attr_method} is not an implemented attribution method."
                )

        eval_stats = _get_evaluation_metrics(model, dloader)
        save_json(eval_stats, experiment_dir / f"{phase_str}_metrics.json")
        np.save(
            experiment_dir / f"{phase_str}_{options.attr_method}_attributions.npy",
            attribution_data,
        )


def _process_genome(
    configs: GenotypeToPhenotypeConfigs, options: ProcessAttributionOptions
):
    for chromosome in tqdm(
        range(1, configs.global_butterfly_metadata.number_of_chromosomes + 1),
        desc="Processing Chromosomes",
        colour="blue",
    ):
        _process_chromosome(configs, options, chromosome=chromosome)


def process_attributions(
    configs: GenotypeToPhenotypeConfigs, options: ProcessAttributionOptions
):
    if configs.experiment.genotype_scope == "genome":
        _process_genome(configs, options)
    elif configs.experiment.genotype_scope == "chromosome":
        _process_chromosome(configs, options)


def process_all_attributions(
    configs: GenotypeToPhenotypeConfigs, options: ProcessAttributionOptions
):
    options = dataclasses.replace(options)

    for species in configs.global_butterfly_metadata.species:
        for wing in configs.global_butterfly_metadata.wings:
            for color in configs.global_butterfly_metadata.phenotypes:
                options.species = species
                options.wing = wing
                options.color = color
                process_attributions(configs=configs, options=options)


@click.command()
@click.option(
    "--configs",
    default=None,
    help="Path to YAML config file to be used in preprocessing.",
)
@ProcessAttributionOptions.click_options()
def main(configs, **kwargs):
    cfgs: GenotypeToPhenotypeConfigs = load_configs(configs)
    opts: ProcessAttributionOptions = ProcessAttributionOptions(**kwargs)

    if opts.process_all:
        process_all_attributions(cfgs, opts)
    else:
        process_attributions(cfgs, opts)


if __name__ == "__main__":
    main()
