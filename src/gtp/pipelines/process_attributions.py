import dataclasses
import os
import torch.multiprocessing as mp
import logging

import click
import numpy as np
import torch
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from gtp.configs.loaders import load_configs
from gtp.configs.project import GenotypeToPhenotypeConfigs
from gtp.dataloading.data_collectors import load_training_data
from gtp.dataloading.datasets import GTP_Dataset, GTP_Individual_Dataset
from gtp.dataloading.path_collectors import get_experiment_directory
from gtp.dataloading.tools import save_json
from gtp.evaluation import (
    AttributionMethod,
    get_lrp_attr,
    get_perturb_attr,
    get_windowed_edit_attr,
)
from gtp.models.net import SoyBeanNet, DeepNet, DeepNet2
from gtp.options.process_attribution import ProcessAttributionOptions
from gtp.tools.calculation import gather_model_predictions_and_actuals
from gtp.utils import setup_logging, create_child_logger

logger = setup_logging(__name__)


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
        attribution_save_path = (
            experiment_dir / f"{phase_str}_{options.attr_method}_attributions.npy"
        )

        # Skip if already calculated
        if not options.force_reprocess and os.path.exists(attribution_save_path):
            continue

        # Below is temporary
        if (
            options.force_reprocess
            and os.path.exists(attribution_save_path)
            and options.attr_method == "windowed_editing"
        ):
            att_data = np.load(attribution_save_path, allow_pickle=True)
            positions = sorted(list(att_data.item().keys()))
            window_size = (positions[1] - positions[0]) // 2
            if window_size == options.window_size:
                continue

        dset = GTP_Dataset(*data)
        dloader = DataLoader(
            dset,
            batch_size=options.batch_size,
            num_workers=options.num_workers,
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
            attribution_save_path,
            attribution_data,
        )


def _process_genome_individuals(
    configs: GenotypeToPhenotypeConfigs, options: ProcessAttributionOptions
):
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

    train_dset, val_dset, test_dset = [
        GTP_Individual_Dataset(*data) for data in load_training_data(configs, options)
    ]

    num_vcfs = train_dset[0][0].shape[1]

    if options.model == "soybean":
        model = SoyBeanNet(
            window_size=num_vcfs,
            num_out_dims=options.out_dims,
            insize=options.insize,
            hidden_dim=options.hidden_dim,
            drop_out_prob=options.drop_out_prob,
        )
    elif options.model == "deepnet":
        model = DeepNet(
            window_size=num_vcfs,
            num_out_dims=options.out_dims,
            insize=options.insize,
            hidden_dim=options.hidden_dim,
            drop_out_prob=options.drop_out_prob,
        ).cuda()

    elif options.model == "deepnet2":
        model = DeepNet2(
            window_size=num_vcfs,
            num_out_dims=options.out_dims,
            insize=options.insize,
        ).cuda()

    model.load_state_dict(torch.load(experiment_dir / "model.pt", weights_only=True))
    model = model.cuda()
    model.eval()

    for dset, phase_str in [
        (train_dset, "training"),
        (val_dset, "validation"),
        (test_dset, "test"),
    ]:
        attribution_save_path = (
            experiment_dir / f"{phase_str}_{options.attr_method}_attributions.npy"
        )

        # Skip if already calculated
        if not options.force_reprocess and os.path.exists(attribution_save_path):
            continue

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
            attribution_save_path,
            attribution_data,
        )


def run_process(cfgs, opts, chromo, queue):
    torch_device = queue.get()
    with torch_device:
        try:
            _process_chromosome(cfgs, opts, chromosome=chromo)
            logger.info(
                f"Completed: {opts.species}-{opts.wing}-{opts.color}-chromosome-{chromo}"
            )
        except Exception as e:
            logger.error(
                f"Exception thrown for {opts.species}-{opts.wing}-{opts.color}-chromosome-{chromo}",
                exc_info=e,
            )

    queue.put(torch_device)


def _process_genome(
    configs: GenotypeToPhenotypeConfigs, options: ProcessAttributionOptions
):
    options.num_workers = 0  # For multi processing to work, we have to set this to 0 for DataLoader. Otherwise, we get a deadlock.

    chromosomes = range(1, configs.global_butterfly_metadata.number_of_chromosomes + 1)
    tbar = tqdm(
        chromosomes,
        desc="Processing Chromosomes",
        colour="blue",
    )
    gpus = [torch.cuda.device(i) for i in range(torch.cuda.device_count())]
    with mp.Manager() as manager:
        queue = manager.Queue()
        for g in gpus:
            queue.put(g)

        configurations = [
            [configs, options, chromosome, queue] for chromosome in chromosomes
        ]

        with mp.Pool(len(gpus)) as p:
            for r in p.starmap(run_process, configurations):
                tbar.update()
                tbar.refresh()


def process_attributions(
    configs: GenotypeToPhenotypeConfigs, options: ProcessAttributionOptions
):
    if configs.experiment.genotype_scope == "genome":
        if options.chromosome == "all":
            _process_genome_individuals(configs, options)
        else:
            _process_genome(configs, options)
    elif configs.experiment.genotype_scope == "chromosome":
        _process_chromosome(configs, options)


def process_all_attributions(
    configs: GenotypeToPhenotypeConfigs, options: ProcessAttributionOptions
):
    options = dataclasses.replace(options)
    progress_logger = create_child_logger(logger, "process_all_attributions")

    for species in configs.global_butterfly_metadata.species:
        for wing in configs.global_butterfly_metadata.wings:
            for color in configs.global_butterfly_metadata.phenotypes:
                options.species = species
                options.wing = wing
                options.color = color
                process_attributions(configs=configs, options=options)
                progress_logger.info(f"Completed: {species}-{wing}-{color}")


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
