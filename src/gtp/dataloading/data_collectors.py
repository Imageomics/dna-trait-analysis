from gtp.configs.project import GenotypeToPhenotypeConfigs
from gtp.dataloading.path_collectors import (
    get_post_processed_genotype_directory,
    get_post_processed_phenotype_directory,
    get_results_training_metadata_directory,
)
from gtp.dataloading.tools import (
    load_chromosome_data,
    split_data_by_file,
)
from gtp.options.training import TrainingOptions
from gtp.tools.timing import profile_exe_time


def load_chromosome_and_phenotype_data(
    configs: GenotypeToPhenotypeConfigs, options: TrainingOptions
):
    genotype_folder = get_post_processed_genotype_directory(configs.io)
    phenotype_folder = get_post_processed_phenotype_directory(configs.io)

    genotype_scope = configs.experiment.genotype_scope
    if options.chromosome == "all":
        genotype_scope = "genome_individuals"

    camids_aligned, genotype_data_aligned, phenotype_data_aligned = (
        load_chromosome_data(
            genotype_folder / genotype_scope,
            phenotype_folder,
            options.species,
            options.wing,
            options.color,
            options.chromosome,
            options.verbose,
        )
    )

    phenotype_data_aligned = phenotype_data_aligned[
        :, options.out_dims_start_idx : options.out_dims_start_idx + options.out_dims
    ]

    return camids_aligned, genotype_data_aligned, phenotype_data_aligned


@profile_exe_time(verbose=False)
def load_training_data(
    configs: GenotypeToPhenotypeConfigs, options: TrainingOptions, return_camids=False
):
    camids_aligned, genotype_data_aligned, phenotype_data_aligned = (
        load_chromosome_and_phenotype_data(configs, options)
    )

    metadata_folder = get_results_training_metadata_directory(configs.io)
    train_split, val_split, test_split = split_data_by_file(
        genotype_data_aligned,
        phenotype_data_aligned,
        camids_aligned,
        metadata_folder,
        options.species,
    )

    if return_camids:
        return train_split, val_split, test_split, camids_aligned

    return train_split, val_split, test_split


@profile_exe_time(verbose=False)
def load_individual_training_data(
    configs: GenotypeToPhenotypeConfigs, options: TrainingOptions, return_camids=False
):
    camids_aligned, genotype_data_aligned, phenotype_data_aligned = (
        load_chromosome_and_phenotype_data(configs, options)
    )

    metadata_folder = get_results_training_metadata_directory(configs.io)
    train_split, val_split, test_split = split_data_by_file(
        genotype_data_aligned,
        phenotype_data_aligned,
        camids_aligned,
        metadata_folder,
        options.species,
    )

    if return_camids:
        return train_split, val_split, test_split, camids_aligned

    return train_split, val_split, test_split
