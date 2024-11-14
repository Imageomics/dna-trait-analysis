import copy
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm
import types

import numpy as np
import pandas as pd

import bitepi

import logging
logging.root.setLevel(logging.DEBUG)

ARGS = types.SimpleNamespace()
ARGS.species = "erato"
ARGS.color = "color_3"
ARGS.wing = "forewings"
#ARGS.genome_folder = "/local/scratch/carlyn.1/dna/vcfs/processed/genome"
#ARGS.phenotype_folder = "/local/scratch/carlyn.1/dna/colors/processed"
ARGS.genome_folder = "/local/scratch/david/geno-pheno-data/dna/processed/genome"
ARGS.phenotype_folder = "/local/scratch/david/geno-pheno-data/colors/processed/"
ARGS.out_dims = 1
ARGS.out_dims_start_idx = 0
ARGS.split_data_folder = "/home/carlyn.1/dna-trait-analysis/data"
ARGS.top_k_chromosome_training_path = "/home/carlyn.1/dna-trait-analysis/plot_results/pvalue_erato_forewings_color_3/top_k_snps_erato_forewings_color_3.npy"



from gtp.dataloading.tools import (
    load_chromosome_data,
    split_data_by_file,
)

def load_data(args):
    camids_aligned, genotype_data_aligned, phenotype_data_aligned = (
        load_chromosome_data(
            args.genome_folder,
            args.phenotype_folder,
            args.species,
            args.wing,
            args.color,
            args.chromosome,
        )
    )

    phenotype_data_aligned = phenotype_data_aligned[
        :, args.out_dims_start_idx : args.out_dims_start_idx + args.out_dims
    ]

    train_split, val_split, test_split = split_data_by_file(
        genotype_data_aligned,
        phenotype_data_aligned,
        camids_aligned,
        args.split_data_folder,
        args.species,
    )

    return train_split, val_split, test_split

def load_one(args, chromosome, snp_idx, idx):
    cur_args = copy.deepcopy(args)
    cur_args.chromosome = chromosome
    train_data, val_data, test_data = load_data(cur_args)
    snp_idx = np.sort(snp_idx).astype(np.int64)
    train_data[0] = train_data[0][:, snp_idx]
    val_data[0] = val_data[0][:, snp_idx]
    test_data[0] = test_data[0][:, snp_idx]
    return (
        idx,
        train_data,
        val_data,
        test_data,
        snp_idx
    )

futures = []
pool = ThreadPoolExecutor(4)
final_train_data = None
final_val_data = None
final_test_data = None
final_snps = []
data = np.load(ARGS.top_k_chromosome_training_path, allow_pickle=True)
test_snp_selections = data.item()["test"]
for idx, snp_idx in tqdm(enumerate(test_snp_selections), desc="loading top snps from chromosome"):
    if idx > 5: break
    future = pool.submit(load_one, ARGS, idx + 1, snp_idx, idx)
    futures.append(future)

total = 0
all_data = []
for future in as_completed(futures):
    proc_idx, train_data, val_data, test_data, snp_idx,  = future.result()
    all_data.append([proc_idx, train_data, val_data, test_data, snp_idx])
    total += 1
    print(f"Completed loading on chromosome: {proc_idx+1}: ({total}/21)")

for proc_idx, train_data, val_data, test_data, snp_idx in sorted(all_data, key=lambda x: int(x[0])):
    print(proc_idx)
    final_snps.extend([f"C_{proc_idx+1}_SNP_{si}" for si in snp_idx])
    if final_train_data is None:
        final_train_data = train_data
        final_val_data = val_data
        final_test_data = test_data
    else:
        final_train_data[0] = np.concatenate(
            (final_train_data[0], train_data[0]), axis=1
        )
        final_val_data[0] = np.concatenate(
            (final_val_data[0], val_data[0]), axis=1
        )
        final_test_data[0] = np.concatenate(
            (final_test_data[0], test_data[0]), axis=1
        )
        assert (train_data[1] == final_train_data[1]).all()
        assert (val_data[1] == final_val_data[1]).all()
        assert (test_data[1] == final_test_data[1]).all()

"""
"0|0": [1, 0, 0], => 0
"1|0": [0, 1, 0], => 1
"0|1": [0, 1, 0], => 1
"1|1": [0, 0, 1], => 2
"""

tx, ty = final_train_data
tx.shape

vx, vy = final_val_data
vx.shape

sample_names = [[f"S{i}", 1] for i in range(tx.shape[0])] + [[f"S{i+tx.shape[0]}", 0] for i in range(vx.shape[0])]
snp_names = final_snps
sample_names

sample_snp_matrix_t = (tx * [0, 1, 2]).sum(-1)
sample_snp_matrix_v = (vx * [0, 1, 2]).sum(-1)
sample_snp_matrix = np.concatenate((sample_snp_matrix_t.T, sample_snp_matrix_v.T), axis=1)

data = np.concatenate((np.array(snp_names)[:, np.newaxis], sample_snp_matrix), axis=1)
sample_array = np.array(sample_names)
df = pd.DataFrame(data, columns=["SNP"] + sample_array[:, 0].tolist())
df.head(4)

# %%
geno_matrix = np.concatenate((df.columns.to_numpy()[np.newaxis, :], df.to_numpy()), axis=0)
geno_matrix = geno_matrix.tolist()
for i in range(len(geno_matrix)):
    for j in range(len(geno_matrix[i])):
        if i == 0: continue
        if j == 0: continue
        geno_matrix[i][j] = int(geno_matrix[i][j])

epistasis = bitepi.Epistasis(
    genotype_array=geno_matrix,
    sample_array=sample_names,
    working_directory="/local/scratch/david/tmp"
)
interactions = epistasis.compute_epistasis(
    #sort=True,
    #best_ig=True,
    threads=2,
    p2=0,
    ig2=0,
)#['best_ig']

print(interactions)


