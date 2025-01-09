from pathlib import Path
from typing import Any

from gtp.configs.io import IOConfigs
from gtp.tools.simple import create_exp_info_text


def _select_override_if_exists(default: Any, override: Any):
    return default if override is None else override


def get_raw_genotype_input_directory(io_cfgs: IOConfigs) -> Path:
    root = _select_override_if_exists(io_cfgs.default_root, io_cfgs.raw_data_input.root)
    path = Path(root, io_cfgs.raw_data_input.genotype)
    return path


def get_raw_phenotype_input_directory(io_cfgs: IOConfigs) -> Path:
    root = _select_override_if_exists(io_cfgs.default_root, io_cfgs.raw_data_input.root)
    path = Path(root, io_cfgs.raw_data_input.phenotype)
    return path


def get_post_processed_genotype_directory(io_cfgs: IOConfigs) -> Path:
    root = _select_override_if_exists(
        io_cfgs.default_root, io_cfgs.data_post_process.root
    )
    path = Path(root, io_cfgs.data_post_process.genotype)
    return path


def get_post_processed_phenotype_directory(io_cfgs: IOConfigs) -> Path:
    root = _select_override_if_exists(
        io_cfgs.default_root, io_cfgs.data_post_process.root
    )
    path = Path(root, io_cfgs.data_post_process.phenotype)
    return path


def get_results_training_metadata_directory(io_cfgs: IOConfigs) -> Path:
    root = _select_override_if_exists(io_cfgs.default_root, io_cfgs.results.root)
    path = Path(root, io_cfgs.results.training_metadata)
    return path


def get_results_training_output_directory(io_cfgs: IOConfigs) -> Path:
    root = _select_override_if_exists(io_cfgs.default_root, io_cfgs.results.root)
    path = Path(root, io_cfgs.results.training_output)
    return path


def get_results_plot_output_directory(io_cfgs: IOConfigs) -> Path:
    root = _select_override_if_exists(io_cfgs.default_root, io_cfgs.results.root)
    path = Path(root, io_cfgs.results.plot_output)
    return path


def get_experiment_directory(
    io_cfgs: IOConfigs,
    species: str,
    wing: str,
    color: str,
    chromosome: int,
    exp_name: str,
) -> Path:
    training_dir = get_results_training_output_directory(io_cfgs)
    exp_info = create_exp_info_text(
        species=species,
        wing=wing,
        color=color,
        chromosome=chromosome,
    )

    experiment_dir = training_dir / exp_name / exp_info
    return experiment_dir
