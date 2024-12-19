import os
import random
from pathlib import Path

import click
import numpy as np

from gtp.configs.loaders import load_configs
from gtp.configs.project import GenotypeToPhenotypeConfigs
from gtp.dataloading.path_collectors import (
    get_post_processed_genotype_directory,
    get_results_training_metadata_directory,
)


def create_training_splits(configs: GenotypeToPhenotypeConfigs, verbose=False):
    for species in configs.global_butterfly_metadata.species:
        # Get camid file. Just need one with the camids in them
        genotype_output_dir = get_post_processed_genotype_directory(configs.io)
        camid_file = None
        for root, _, files in os.walk(
            genotype_output_dir / configs.experiment.genotype_scope / species
        ):
            for f in files:
                if f == "camids.npy":
                    camid_file = Path(root) / f
                    break
            if camid_file is not None:
                break

        # Load CAMID file
        camids = np.load(camid_file).tolist()

        # Shuffle CAMIDs
        random.seed(configs.training.seed)
        random.shuffle(camids)

        # Create Splits
        train_idx = int(len(camids) * configs.training.train_ratio)
        val_idx = int(len(camids) * configs.training.validation_ratio)

        train_split = camids[:train_idx]
        val_split = camids[train_idx : train_idx + val_idx]
        test_split = camids[train_idx + val_idx :]

        if verbose:
            print(f"{species.capitalize()} training data length: {len(train_split)}")
            print(f"{species.capitalize()} validation data length: {len(val_split)}")
            print(f"{species.capitalize()} testing data length: {len(test_split)}")

        # Save splits
        training_metadata_dir = get_results_training_metadata_directory(configs.io)
        os.makedirs(training_metadata_dir, exist_ok=True)
        np.save(
            Path(training_metadata_dir, f"{species}_train.npy"),
            train_split,
        )
        np.save(Path(training_metadata_dir, f"{species}_val.npy"), val_split)
        np.save(
            Path(training_metadata_dir, f"{species}_test.npy"),
            test_split,
        )


@click.command()
@click.option(
    "--configs",
    default=None,
    help="Path to YAML config file to be used in preprocessing.",
)
@click.option(
    "--verbose/--no-verbose", default=False, help="Whether or not to see logging"
)
def main(configs, verbose):
    cfgs: GenotypeToPhenotypeConfigs = load_configs(configs)
    create_training_splits(configs=cfgs, verbose=verbose)


if __name__ == "__main__":
    main()
