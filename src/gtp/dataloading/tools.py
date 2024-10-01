import os
from collections import defaultdict

import numpy as np


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
