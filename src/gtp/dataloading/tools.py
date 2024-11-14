import os
from collections import defaultdict

import numpy as np
import pandas as pd

from gtp.tools import profile_exe_time


# Deprecate soon
def align_data(dna_data, dna_camids, pheno_data):
    # Remove Extra DNA data
    idx_to_rm = []
    for i, camid in enumerate(dna_camids):
        if camid not in pheno_data.camid.values.tolist():
            idx_to_rm.append(i)
    for rmi in reversed(idx_to_rm):
        dna_data = np.delete(dna_data, rmi, axis=0)
        dna_camids = np.delete(dna_camids, rmi, axis=0)

    # Remove Extra Phenotype data
    idx_to_rm = []
    for i, camid in enumerate(pheno_data.camid):
        if camid not in dna_camids.tolist():
            idx_to_rm.append(i)
    for rmi in reversed(idx_to_rm):
        pheno_data.drop(index=rmi)

    # Sort
    pheno_data.sort_values(
        by="camid",
        key=lambda column: column.map(lambda e: dna_camids.tolist().index(e)),
        inplace=True,
    )

    assert (
        dna_camids.shape[0] == pheno_data.shape[0] == dna_data.shape[0]
    ), "Unequal X and Y in data"
    for x, y in zip(pheno_data.camid.values.tolist(), dna_camids.tolist()):
        assert x == y, f"{x} != {y}. Data not aligned"

    return dna_data, dna_camids, pheno_data


@profile_exe_time
def collect_chromosome(root, species, chromosome):
    genotype_path_root = os.path.join(root, f"{species}")
    scaffolds = []
    for subdir in os.listdir(genotype_path_root):
        if species == "erato":
            chrom_info = subdir.replace("Herato", "")
            chrom_num = int(chrom_info[:2])
            scaf_num = int(chrom_info[2:])
        elif species == "melpomene":
            chrom_info = subdir.replace("Hmel2", "")
            chrom_num = int(chrom_info[:2])
            scaf_num = int(chrom_info[2:-1])
        else:
            assert False, f"Invalid species: {species}"

        if chrom_num == chromosome:
            scaffolds.append([subdir, scaf_num])
    scaffolds = sorted(scaffolds, key=lambda x: x[1])

    final_camids = None
    compiled = None
    for scaffold_dir, _ in scaffolds:
        tgt_dir = os.path.join(genotype_path_root, scaffold_dir)
        camids = np.load(os.path.join(tgt_dir, "camids.npy"))
        data = np.load(os.path.join(tgt_dir, "ml_ready.npy"))
        sort_idx = np.argsort(camids)
        sorted_data = data[sort_idx].astype(np.byte)
        sorted_camids = camids[sort_idx]
        if final_camids is not None:
            assert (final_camids == sorted_camids).all(), "INVALID SORT"
            compiled = np.concatenate((compiled, sorted_data), axis=1)
        else:
            final_camids = sorted_camids
            compiled = sorted_data

    return final_camids, compiled


@profile_exe_time
def load_phenotype_data(phenotype_folder, species, wing, color):
    """Loads phenotype data for Heliconius butterflies (Erato & Melpomene)

    Args:
        phenotype_folder (_type_): path to phenotype folder
        species (_type_): species to collect (erato or melpomene)
        wing (_type_): which wingside traits to collect (forewings or hindwings)
        color (_type_): which color traits to collect (color_1, color_2, color_3, total)

    Returns:
        _type_: a list of camids for the specimens and the phenotype data
    """
    pca_path = os.path.join(phenotype_folder, f"{species}_{wing}_{color}", "data.csv")
    pca_df = pd.read_csv(pca_path)
    pca_camids = pca_df.camid.to_numpy()
    pca_data = pca_df.iloc[:, 1:].to_numpy()  # ignore the camid column

    return pca_camids, pca_data


@profile_exe_time
def align_genotype_and_phenotype_data(
    phenotype_camids, genotype_camids, phenotype_data, genotype_data
):
    """Aligns genotype and phenotype data by their camids since they aren't a perfect match originally.

    Args:
        phenotype_camids (np.ndarray): camids of the phenotype data (must be aligned with the phenotype data)
        genotype_camids (np.ndarray): camids of the genotype data (must be aligned with the genotype data)
        phenotype_data (np.ndarray): Phenotype data
        genotype_data (np.ndarray): Genotype data

    Returns:
        _type_: Aligned phenotype_data, genotype_data, camids
    """
    conflict_camid_a = list(
        set.difference(set(phenotype_camids.tolist()), set(genotype_camids.tolist()))
    )
    conflict_camid_b = list(
        set.difference(set(genotype_camids.tolist()), set(phenotype_camids.tolist()))
    )
    conflict_camids = conflict_camid_a + conflict_camid_b

    idx = np.isin(genotype_camids, conflict_camids)
    genotype_camids_aligned = genotype_camids[~idx]
    genotype_data_aligned = genotype_data[~idx]

    idx = np.isin(phenotype_camids, conflict_camids)
    phenotype_camids_aligned = phenotype_camids[~idx]
    phenotype_data_aligned = phenotype_data[~idx]

    assert (
        phenotype_camids_aligned == genotype_camids_aligned
    ).all(), "Invalid alignment"

    return phenotype_data_aligned, genotype_data_aligned, genotype_camids_aligned


@profile_exe_time
def load_chromosome_data(
    genotype_folder, phenotype_folder, species, wing, color, chromosome, verbose=False
):
    """
    NOTE: There are missing camids from either pheno or geno type data
    Seems to consistently be 1 missing from melpomene genotype
    and 4 missing from erato phenotype
    """

    def print_out(str):
        if verbose:
            print(str)

    # Collect phenotype data
    pca_camids, pca_data = load_phenotype_data(phenotype_folder, species, wing, color)

    # Collect genotype data
    genotype_camids, genotype_data = collect_chromosome(
        genotype_folder, species, chromosome
    )

    # Align data
    phenotype_data_aligned, genotype_data_aligned, camids_aligned = (
        align_genotype_and_phenotype_data(
            pca_camids, genotype_camids, pca_data, genotype_data
        )
    )

    return camids_aligned, genotype_data_aligned, phenotype_data_aligned


def split_data_by_file(
    genotype_data, phenotype_data, camids, split_data_folder, species
):
    """Takes the loaded data and splits into train, val, test according to file by camid

    Args:
        genotype_data (_type_): Genotype data
        phenotype_data (_type_): Phenotype data
        camids (_type_): list of aligned camids with genotype and phenotype data
        split_data_folder (_type_): path to folder with data split file
        species (_type_): butterfly species (erato or melpomene)

    Returns:
        _type_: train, val, and test splits with genotype and phenotype data
    """
    train_cams = np.load(os.path.join(split_data_folder, f"{species}_train.npy"))
    val_cams = np.load(os.path.join(split_data_folder, f"{species}_val.npy"))
    test_cams = np.load(os.path.join(split_data_folder, f"{species}_test.npy"))

    train_idx = np.isin(camids, train_cams)
    train_phenotype_data = phenotype_data[train_idx]
    train_geno_data = genotype_data[train_idx]

    val_idx = np.isin(camids, val_cams)
    val_phenotype_data = phenotype_data[val_idx]
    val_geno_data = genotype_data[val_idx]

    test_idx = np.isin(camids, test_cams)
    test_phenotype_data = phenotype_data[test_idx]
    test_geno_data = genotype_data[test_idx]

    return (
        [train_geno_data, train_phenotype_data],
        [val_geno_data, val_phenotype_data],
        [test_geno_data, test_phenotype_data],
    )


def butterfly_states_to_ml_ready(df):
    # 0:
    # 1:
    state_map = defaultdict(
        lambda: [0, 0, 0],
        {
            "0|0": [1, 0, 0],
            "1|0": [0, 1, 0],
            "0|1": [0, 1, 0],
            "1|1": [0, 0, 1],
        },
    )

    ml_ready = df.map(lambda x: state_map[x])
    ml_ready = np.array(ml_ready.values.tolist())

    return ml_ready
