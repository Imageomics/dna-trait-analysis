import dataclasses
import os
from enum import Enum
from typing import Any

import click
import dash_bio as dashbio
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
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

sns.set_theme()


class Phases(Enum):
    TRAINING = "training"
    VALIDATION = "validation"
    TESTING = "testing"


def _process_chromosome(
    configs: GenotypeToPhenotypeConfigs,
    options: PlotAttributionOptions,
    chromosome: int = None,
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
    if not experiment_dir.exists():
        raise FileNotFoundError(
            f"{experiment_dir} does not exist. Unable to process chromosome {options.chromosome}"
        )

    # Load attributions
    train_attributions = np.load(
        experiment_dir / f"training_{options.attr_method}_attributions.npy"
    )
    val_attributions = np.load(
        experiment_dir / f"validation_{options.attr_method}_attributions.npy"
    )
    test_attributions = np.load(
        experiment_dir / f"test_{options.attr_method}_attributions.npy"
    )

    # Aggreate attributions
    plot_targets = [
        idx + options.out_dims_start_idx_attribution
        for idx in range(options.out_dims_attribution)
    ]
    if options.attribution_aggregation == "mean":
        train_attributions = train_attributions[plot_targets].mean(0)
        val_attributions = val_attributions[plot_targets].mean(0)
        test_attributions = test_attributions[plot_targets].mean(0)
    elif options.attribution_aggregation == "sum":
        train_attributions = train_attributions[plot_targets].sum(0)
        val_attributions = val_attributions[plot_targets].sum(0)
        test_attributions = test_attributions[plot_targets].sum(0)
    elif options.attribution_aggregation == "max":
        train_attributions = np.max(train_attributions[plot_targets], axis=0)
        val_attributions = np.max(val_attributions[plot_targets], axis=0)
        test_attributions = np.max(test_attributions[plot_targets], axis=0)
    else:
        raise NotImplementedError(
            f"{options.attribution_aggregation} has not been implemented as an aggregation method."
        )

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

    # ["BP", "CHR", "SNP", "Attribution"]
    attribution_idx = 3

    if options.top_n > 0:
        # Order and filter by attributions
        df_train_data = df_train_data[
            df_train_data[:, attribution_idx].astype(np.float32).argsort()
        ][-options.top_n :]
        df_val_data = df_val_data[
            df_val_data[:, attribution_idx].astype(np.float32).argsort()
        ][-options.top_n :]
        df_test_data = df_test_data[
            df_test_data[:, attribution_idx].astype(np.float32).argsort()
        ][-options.top_n :]

        # Reorder by base pair
        df_train_data = df_train_data[df_train_data[:, 0].astype(np.int64).argsort()]
        df_val_data = df_val_data[df_val_data[:, 0].astype(np.int64).argsort()]
        df_test_data = df_test_data[df_test_data[:, 0].astype(np.int64).argsort()]

    column_names = ["BP", "CHR", "SNP", "Attribution"]
    # NOTE: The PCC (pearson correlation coeficient) and the PVAL (pvalue) are measured between the attribution the ground truth phenotype.
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


def _plot_manhattan_plot_static_matplotlib(
    configs: GenotypeToPhenotypeConfigs,
    options: PlotAttributionOptions,
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    metrics: Any,
    y_val: str,
):
    # Plot constants
    TITLE_FONT_SIZE = 22
    SUB_TITLE_FONT_SIZE = 18
    AXIS_FONT_SIZE = 16
    BAR_TOP_TEXT_SIZE = 6
    EPSILON = 1e-20

    colors = ["grey", "black"]
    fig, axs = plt.subplots(3, 1, figsize=(16, 18))
    title_str = (
        f"{options.species.capitalize()} ({options.wing}) | Phenotype: {options.color}"
    )
    fig.suptitle(title_str, fontsize=TITLE_FONT_SIZE)

    if y_val in ["PVAL"]:  # TODO handle 0 pvalues....
        new_y_val = f"-log10({y_val})"
        train_df[new_y_val] = -np.log10(train_df[y_val])
        val_df[new_y_val] = -np.log10(val_df[y_val])
        test_df[new_y_val] = -np.log10(test_df[y_val])
        y_val = new_y_val
    if y_val in ["Attribution", "PCC"]:
        max_attribution_score = max(
            train_df[y_val].max(),
            val_df[y_val].max(),
            test_df[y_val].max(),
        )
    else:
        raise NotImplementedError(
            f"{y_val} is not a valid y_val for manhattan plotting"
        )

    bar_value_key = "pvalue"

    def do_neg_log_on_bar_y():
        return bar_value_key in ["pvalue", "rmse"]

    max_bar_height = max(
        metrics[phase][chr][bar_value_key]
        for phase in list(Phases)
        for chr in metrics[phase]
    )
    if do_neg_log_on_bar_y():
        minval = min(
            metrics[phase][chr][bar_value_key]
            for phase in list(Phases)
            for chr in metrics[phase]
        )
        minval = EPSILON if minval == 0.0 else minval
        max_bar_height = -np.log10(minval)

    for i, (df, phase) in enumerate(
        [
            (train_df, Phases.TRAINING),
            (val_df, Phases.VALIDATION),
            (test_df, Phases.TESTING),
        ]
    ):
        phase_metrics = metrics[phase]
        bar_chart_ax = axs[i]
        manhattan_plot_ax = axs[i].twinx()

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

            bar_y = phase_metrics[chromosome][bar_value_key]

            if bar_value_key in ["pvalue", "rmse"]:
                bar_y = EPSILON if bar_y == 0.0 else bar_y
                bar_y = -np.log10(bar_y)

            bar = bar_chart_ax.bar(
                x_lbl_pos,
                bar_y,
                width=group["idx"].iloc[-1] - group["idx"].iloc[0],
                color="lightcoral",
                alpha=0.3,
                edgecolor="maroon",
                linewidth=1,
            )

            # Add RMSE to top of bar
            bar_height = bar.patches[0].get_height()
            bar_width = bar.patches[0].get_width()
            bar_chart_ax.text(
                (bar.patches[0].get_x() + bar_width / 2),
                bar_height + 0.1 * max_bar_height,
                f"RMSE\n{round(phase_metrics[chromosome]['rmse'], 2)}",
                fontsize=BAR_TOP_TEXT_SIZE,
                horizontalalignment="center",
                verticalalignment="center",
            )

            group.plot(
                kind="scatter",
                x="idx",
                y=y_val,
                color=colors[chromosome % len(colors)],
                ax=manhattan_plot_ax,
            )
        manhattan_plot_ax.set_xticks(x_lbls_pos)
        manhattan_plot_ax.set_xticklabels(x_lbls)

        # set axis limits
        manhattan_plot_ax.set_xlim([0, len(df)])
        # _, manhattan_max_y_display_value = manhattan_plot_ax.transData.transform(
        #     (0, max_attribution_score)
        # )
        # _, bar_max_y_display_value = bar_chart_ax.transData.transform(
        #     (0, max_bar_height)
        # )
        manhattan_plot_ax.set_ylim(
            [0, max_attribution_score + 50]
        )  # add 50 as a visual buffer
        bar_chart_ax.set_ylim([0, max_bar_height * 1.2])

        # axis label
        manhattan_plot_ax.set_xlabel("Chromosome", fontsize=AXIS_FONT_SIZE)
        manhattan_plot_ax.set_ylabel(f"{y_val} (points)", fontsize=AXIS_FONT_SIZE)
        bar_y_label = (
            f"{bar_value_key.upper()}"
            if not do_neg_log_on_bar_y()
            else f"-log_10({bar_value_key.upper()})"
        )
        bar_chart_ax.set_ylabel(
            f"{bar_y_label} (bar)", color="red", fontsize=AXIS_FONT_SIZE
        )

        # Title
        manhattan_plot_ax.set_title(
            phase.value.capitalize(), fontsize=SUB_TITLE_FONT_SIZE
        )

        # Remove grid lines
        manhattan_plot_ax.grid(False)
        bar_chart_ax.grid(False)

    return fig


def create_manhattan_plot_static_matplotlib(
    configs: GenotypeToPhenotypeConfigs, options: PlotAttributionOptions
):
    if (
        configs.experiment.genotype_scope == "genome"
        and not options.plot_one_chromosome
    ):
        train_df, val_df, test_df, metrics = _process_genome(configs, options)
    elif (
        configs.experiment.genotype_scope == "chromosome" or options.plot_one_chromosome
    ):
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
    else:
        raise NotImplementedError("Plot configurations are not valid.")

    for tgt_y_val in ["Attribution"]:
        fig = _plot_manhattan_plot_static_matplotlib(
            configs=configs,
            options=options,
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            metrics=metrics,
            y_val=tgt_y_val,
        )

        plot_output_dir = get_results_plot_output_directory(configs.io)
        plot_output_dir_final = (
            plot_output_dir
            / options.exp_name
            / f"{options.species}_{options.wing}_{options.color}"
        )
        plot_output_dir_final.mkdir(exist_ok=True, parents=True)

        figure_path = plot_output_dir_final / f"manhattan_plot_{tgt_y_val}.png"
        fig.savefig(figure_path)
        print(f"Figure saved at: {figure_path}")
        plt.close(fig)


def plot(configs: GenotypeToPhenotypeConfigs, options: PlotAttributionOptions):
    create_manhattan_plot_static_matplotlib(configs, options)


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
