import os
import json
from argparse import ArgumentParser
from tqdm import tqdm

import numpy as np

from data_tools import parse_vcfs, get_data_matrix_from_vcfs

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--vcf", type=str, default="/local/scratch/carlyn.1/dna/vcfs/erato/Herato1001_wntA.tsv")
    parser.add_argument("--output_dir", type=str, default="/local/scratch/carlyn.1/dna/vcfs/")

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()
    base_name = os.path.splitext(os.path.basename(args.vcf))[0]
    vcfs = parse_vcfs(args.vcf)
    
    specimens = {}
    for vcf in vcfs:
        specimen_name = vcf.specimen
        if specimen_name not in specimens:
            specimens[specimen_name] = []
        specimens[specimen_name].append(vcf)

    specimen_data = []
    metadata = []
    for specimen in tqdm(specimens, desc=f"Preping VCFs"):
        specimen_data_ex = specimens[specimen]
        data = get_data_matrix_from_vcfs(specimen_data_ex)
        specimen_data.append(data)
        metadata.append(specimen_data_ex[0].specimen)
    
    save_json(os.path.join(args.output_dir, f"{base_name}_names.json"), metadata)
    np.savez(os.path.join(args.output_dir, f"{base_name}_vcfs.npz"), np.array(specimen_data))
    