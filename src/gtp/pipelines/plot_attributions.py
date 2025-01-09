import dataclasses
import os
from enum import Enum

import click
import dash_bio as dashbio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from gtp.configs.loaders import load_configs
from gtp.configs.project import GenotypeToPhenotypeConfigs
from gtp.dataloading.path_collectors import (
    get_post_processed_genotype_directory,
    get_results_plot_output_directory,
    get_results_training_output_directory,
)
from gtp.dataloading.tools import collect_chromosome_position_metadata, load_json
from gtp.options.plot_attribution import PlotAttributionOptions
from gtp.tools.simple import create_exp_info_text


class Phases(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


def _process_chromosome(
    configs: GenotypeToPhenotypeConfigs,
    options: PlotAttributionOptions,
    chromosome: int,
):
    if chromosome is not None:
        options = dataclasses.replace(options)
        options.chromosome = chromosome

    training_output_dir = get_results_training_output_directory(configs.io)
    exp_info = create_exp_info_text(
        species=options.species,
        wing=options.wing,
        color=options.color,
        chromosome=options.chromosome,
    )

    experiment_dir = training_output_dir / options.exp_name / exp_info
    if not os.path.exists(experiment_dir):
        print(
            f"{experiment_dir} does not exist. Unable to process chromosome {options.chromosome}"
        )
        return

    # Load attributions
    train_attributions = np.load(experiment_dir / "training_attributions.npy")
    val_attributions = np.load(experiment_dir / "validation_attributions.npy")
    test_attributions = np.load(experiment_dir / "test_attributions.npy")
    train_metrics = load_json(experiment_dir / "training_metrics.json")
    val_metrics = load_json(experiment_dir / "validation_metrics.json")
    test_metrics = load_json(experiment_dir / "test_metrics.json")

    # Get Chromosome metadata
    genotype_folder = get_post_processed_genotype_directory(configs.io)

    position_metadata = collect_chromosome_position_metadata(
        genotype_folder / configs.experiment.genotype_scope,
        options.species,
        options.chromosome,
    )

    assert (
        len(position_metadata)
        == train_attributions.shape[0]
        == val_attributions.shape[0]
        == test_attributions.shape[0]
    ), "Must have matching length to be accurate."

    # Creating metadata values [[{CHROMOSOME}, '{SCAFFOLD}:{SCAFFOLD_POSITION}"], ...]
    metadata = [
        [i, int(options.chromosome), f"{x[0]}:{x[1]}"]
        for i, x in enumerate(position_metadata)
    ]

    df_train_data = np.concatenate(
        (metadata, train_attributions[:, np.newaxis]), axis=1
    )
    df_val_data = np.concatenate((metadata, val_attributions[:, np.newaxis]), axis=1)
    df_test_data = np.concatenate((metadata, test_attributions[:, np.newaxis]), axis=1)

    if options.top_n > 0:
        # Order and filter by attributions
        df_train_data = df_train_data[
            df_train_data[:, -1].astype(np.float32).argsort()
        ][-options.top_n :]
        df_val_data = df_val_data[df_val_data[:, -1].astype(np.float32).argsort()][
            -options.top_n :
        ]
        df_test_data = df_test_data[df_test_data[:, -1].astype(np.float32).argsort()][
            -options.top_n :
        ]

        # Reorder by base pair
        df_train_data = df_train_data[df_train_data[:, 0].astype(np.int64).argsort()]
        df_val_data = df_val_data[df_val_data[:, 0].astype(np.int64).argsort()]
        df_test_data = df_test_data[df_test_data[:, 0].astype(np.int64).argsort()]

    column_names = ["BP", "CHR", "SNP", "Attribution"]
    train_df = pd.DataFrame(data=df_train_data, columns=column_names)
    val_df = pd.DataFrame(data=df_val_data, columns=column_names)
    test_df = pd.DataFrame(data=df_test_data, columns=column_names)
    casting_kwargs = {"CHR": "int64", "BP": "int32", "Attribution": "float32"}
    train_df = train_df.astype(casting_kwargs)
    val_df = val_df.astype(casting_kwargs)
    test_df = test_df.astype(casting_kwargs)

    return train_df, val_df, test_df, train_metrics, val_metrics, test_metrics


def _process_genome(
    configs: GenotypeToPhenotypeConfigs, options: PlotAttributionOptions
):
    train_dfs = []
    val_dfs = []
    test_dfs = []
    metrics = {
        Phases.TRAINING: {},
        Phases.VALIDATION: {},
        Phases.TESTING: {},
    }
    for chromosome in tqdm(
        range(1, configs.global_butterfly_metadata.number_of_chromosomes + 1),
        desc="Processing Chromosomes",
        colour="blue",
    ):
        trdf, vdf, tedf, trm, vm, tem = _process_chromosome(
            configs, options, chromosome=chromosome
        )
        train_dfs.append(trdf)
        val_dfs.append(vdf)
        test_dfs.append(tedf)
        metrics[Phases.TRAINING][chromosome] = trm
        metrics[Phases.VALIDATION][chromosome] = vm
        metrics[Phases.TESTING][chromosome] = tem

        if chromosome >= 3 and options.run_test:
            break

    return pd.concat(train_dfs), pd.concat(val_dfs), pd.concat(test_dfs), metrics


def create_manhattan_plot_static(
    configs: GenotypeToPhenotypeConfigs, options: PlotAttributionOptions
):
    if configs.experiment.genotype_scope == "genome":
        val_df, test_df = _process_genome(configs, options)
    elif configs.experiment.genotype_scope == "chromosome":
        # There is a bug if we just process a single chromosome in the dashbio package
        val_df, test_df = _process_chromosome(configs, options)

    plot_kwargs = {
        "p": "Attribution",
        "chrm": "CHR",
        "bp": "BP",
        "snp": "SNP",
        "gene": None,
        "logp": False,
        "ylabel": "Model Attribution Score",
        "highlight": False,
    }

    title_str = (
        f"{options.species.capitalize()} ({options.wing}) | Phenotype: {options.color}"
    )

    manhattanplot = dashbio.ManhattanPlot(
        dataframe=val_df,
        highlight_color="#00FFAA",
        title=title_str,
        **plot_kwargs,
    )

    return manhattanplot


def create_manhattan_plot_static_matplotlib(
    configs: GenotypeToPhenotypeConfigs, options: PlotAttributionOptions
):
    if configs.experiment.genotype_scope == "genome":
        train_df, val_df, test_df, metrics = _process_genome(configs, options)
    elif configs.experiment.genotype_scope == "chromosome":
        # There is a bug if we just process a single chromosome in the dashbio package
        train_df, val_df, test_df, train_metrics, val_metrics, test_metrics = (
            _process_chromosome(configs, options)
        )
        metrics = {
            Phases.TRAINING: {
                options.chromosome: train_metrics,
            },
            Phases.VALIDATION: {
                options.chromosome: val_metrics,
            },
            Phases.TESTING: {
                options.chromosome: test_metrics,
            },
        }

    colors = ["grey", "black"]
    fig, axs = plt.subplots(3, 1, figsize=(14, 16))
    for i, (df, phase) in enumerate(
        [
            (train_df, Phases.TRAINING),
            (val_df, Phases.VALIDATION),
            (test_df, Phases.TESTING),
        ]
    ):
        phase_metrics = metrics[phase]
        bar_chart_ax = axs[i].twinx()
        bar_chart_ax.set_ylabel("Pearson R", color="red")

        df["idx"] = range(len(df))
        df_grouped = df.groupby(("CHR"))
        x_lbls = []
        x_lbls_pos = []
        for name, group in df_grouped:
            chromosome = group.iloc[0]["CHR"]
            x_lbls.append(name)
            x_lbl_pos = (
                group["idx"].iloc[-1]
                - (group["idx"].iloc[-1] - group["idx"].iloc[0]) / 2
            )
            x_lbls_pos.append(x_lbl_pos)

            bar_chart_ax.bar(
                x_lbl_pos,
                phase_metrics[chromosome]["pearsonr"],
                width=group["idx"].iloc[-1] - group["idx"].iloc[0],
                zorder=0,
                color="lightcoral",
            )
            group.plot(
                kind="scatter",
                x="idx",
                y="Attribution",
                color=colors[chromosome % len(colors)],
                ax=axs[i],
                zorder=10,
            )
        axs[i].set_xticks(x_lbls_pos)
        axs[i].set_xticklabels(x_lbls)

        # set axis limits
        axs[i].set_xlim([0, len(df)])
        bar_chart_ax.set_ylim([0, 1])

        # x axis label
        axs[i].set_xlabel("Chromosome")

        # Title
        title_str = f"{options.species.capitalize()} ({options.wing}) | Phenotype: {options.color} ({phase.value})"
        axs[i].set_title(title_str)

    return fig


def plot(configs: GenotypeToPhenotypeConfigs, options: PlotAttributionOptions):
    fig = create_manhattan_plot_static_matplotlib(configs, options)
    plot_output_dir = get_results_plot_output_directory(configs.io)
    plot_output_dir_final = (
        plot_output_dir
        / options.exp_name
        / f"{options.species}_{options.wing}_{options.color}"
    )
    plot_output_dir_final.mkdir(exist_ok=True, parents=True)

    figure_path = plot_output_dir_final / "manhattan_plot.png"
    fig.savefig(figure_path)
    print(f"Figure saved at: {figure_path}")
    plt.close(fig)


def plot_all(configs: GenotypeToPhenotypeConfigs, options: PlotAttributionOptions):
    options = dataclasses.replace(options)

    for species in configs.global_butterfly_metadata.species:
        for wing in configs.global_butterfly_metadata.wings:
            for color in configs.global_butterfly_metadata.phenotypes:
                options.species = species
                options.wing = wing
                options.color = color
                plot(configs=configs, options=options)


@click.command()
@click.option(
    "--configs",
    default=None,
    help="Path to YAML config file to be used in preprocessing.",
)
@PlotAttributionOptions.click_options()
def main(configs, **kwargs):
    cfgs: GenotypeToPhenotypeConfigs = load_configs(configs)
    opts: PlotAttributionOptions = PlotAttributionOptions(**kwargs)

    if opts.process_all:
        plot_all(cfgs, opts)
    else:
        plot(cfgs, opts)


if __name__ == "__main__":
    main()
