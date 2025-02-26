import click
from tqdm.auto import tqdm

from gtp.configs.loaders import load_configs
from gtp.configs.project import GenotypeToPhenotypeConfigs


def preprocess_phenotypes(configs: GenotypeToPhenotypeConfigs, verbose=False):
    from gtp.dataloading.data_preprocessors import ButterflyPatternizePreprocessor
    from gtp.dataloading.path_collectors import (
        get_post_processed_phenotype_directory,
        get_raw_phenotype_input_directory,
    )

    phenotype_input_dir = get_raw_phenotype_input_directory(configs.io)
    phenotype_output_dir = get_post_processed_phenotype_directory(configs.io)

    preprocessor = ButterflyPatternizePreprocessor(
        input_dir=phenotype_input_dir, output_dir=phenotype_output_dir, verbose=verbose
    )

    preprocessing_options = []
    for species in configs.global_butterfly_metadata.species:
        for wing in configs.global_butterfly_metadata.wings:
            for phenotype in configs.global_butterfly_metadata.phenotypes:
                preprocessing_options.append([species, wing, phenotype])

    for species, wing, phenotype in tqdm(
        preprocessing_options,
        desc="Preprocessing Phenotypes",
        colour="blue",
    ):
        suffix_path = f"{species}_{wing}_PCA/PCA_{phenotype}_loadings.csv"
        preprocessor.process(pca_csv_path_suffix=suffix_path)
        preprocessor.save_result(f"{species}_{wing}_{phenotype}")


def _genotype_preprocess_fn(process_item):
    from gtp.tools.simple import convert_bytes

    file_size, _species, pca_csv_path_suffix, save_dir, preprocessor, verbose, process_max_rows = (
        process_item
    )
    if verbose:
        print(f"Processing {pca_csv_path_suffix}: {convert_bytes(file_size)} bytes")
    try:            
        preprocessor.process(pca_csv_path_suffix=pca_csv_path_suffix, process_max_rows=process_max_rows)
        preprocessor.save_result(save_dir)
    except Exception as e:
        print(e)
        return False

    return True


def preprocess_genotypes(
    configs: GenotypeToPhenotypeConfigs,
    force_reprocess: bool = False,
    num_processes: int = 4,
    verbose: bool = False,
):
    import os
    from multiprocessing import Pool
    from pathlib import Path

    from tqdm.auto import tqdm

    from gtp.dataloading.data_preprocessors import ButterflyGenePreprocessor
    from gtp.dataloading.path_collectors import (
        get_post_processed_genotype_directory,
        get_raw_genotype_input_directory,
    )

    genotype_input_dir = get_raw_genotype_input_directory(configs.io)
    genotype_output_dir = get_post_processed_genotype_directory(configs.io)
    preprocessor = ButterflyGenePreprocessor(
        input_dir=genotype_input_dir,
        output_dir=genotype_output_dir,
        verbose=verbose,
    )
    
    # Process a subset of the genome for testing
    process_max_rows = None
    if configs.experiment.do_subset:
        process_max_rows = 1000
    
    process_data = []

    # Collect data to be processed
    for species in configs.global_butterfly_metadata.species:
        species_genome_path = Path(f"{species}/{configs.experiment.genotype_scope}")
        for root, dirs, files in os.walk(genotype_input_dir / species_genome_path):
            print(root)
            for i, f in enumerate(files):
                fname = f.split(".")[0]

                # Skip if already processed and not being forced to reprocess
                if not force_reprocess and os.path.exists(
                    genotype_output_dir
                    / f"{configs.experiment.genotype_scope}/{species}/{fname}/ml_ready.npy"
                ):
                    continue

                # Gather information
                genome_file_path = species_genome_path / f
                size = os.path.getsize(genotype_input_dir / genome_file_path)
                process_data.append(
                    [
                        size,
                        species,
                        genome_file_path,
                        f"{configs.experiment.genotype_scope}/{species}/{fname}",
                    ]
                )

    # Sort by file size. Process smaller files first
    process_data = sorted(process_data, key=lambda x: (x[1], x[0]))
    process_data = [
        x + [preprocessor, verbose, process_max_rows] for x in process_data
    ]  # Append preprocessor and verbose variables for compatibility for multiprocessing function

    with (
        Pool(processes=num_processes) as p,
        tqdm(
            total=len(process_data),
            desc="Processing Genotype data",
            colour="#87ceeb",  # Skyblue
        ) as pbar,
    ):
        for result in p.imap_unordered(_genotype_preprocess_fn, process_data):
            pbar.update()
            pbar.refresh()


@click.command()
@click.option(
    "--configs",
    default=None,
    help="Path to YAML config file to be used in preprocessing.",
)
@click.option(
    "--method",
    default="both",
    type=click.Choice(["phenotype", "genotype", "both"]),
    prompt="Select preprocessing method to run:",
    help="Which preprocessing method to run [phenotype, genotype, both]",
)
@click.option(
    "--verbose/--no-verbose", default=False, help="Whether or not to see logging"
)
@click.option(
    "--force-reprocess/--no-force-reprocess",
    default=False,
    help="Whether or not to force reprocessing",
)
@click.option(
    "--num-processes",
    default=4,
    help="Number of processes to use in processing genotypes.",
)
def main(configs, method, force_reprocess, num_processes, verbose):
    cfgs: GenotypeToPhenotypeConfigs = load_configs(configs)
    if method in ["phenotype", "both"]:
        preprocess_phenotypes(cfgs, verbose)
    if method in ["genotype", "both"]:
        preprocess_genotypes(cfgs, force_reprocess, num_processes, verbose)


if __name__ == "__main__":
    main()
