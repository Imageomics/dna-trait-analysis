import os
import json

import numpy as np

from data_tools import parse_vcfs, get_data_matrix_from_vcfs

def save_json(path, data):
    with open(path, 'w') as f:
        json.dump(data, f)

if __name__ == "__main__":
    vcfs = parse_vcfs("/local/scratch/carlyn.1/dna/vcfs/erato_optix_variants.tsv")
    OUTPUT = "/local/scratch/carlyn.1/dna/vcfs/"
    
    specimens = {}
    for vcf in vcfs:
        specimen_name = vcf.specimen
        if specimen_name not in specimens:
            specimens[specimen_name] = []
        specimens[specimen_name].append(vcf)

    specimen_data = []
    metadata = []
    for specimen in specimens:
        specimen_data_ex = specimens[specimen]
        data = get_data_matrix_from_vcfs(specimen_data_ex)
        specimen_data.append(data)
        metadata.append(specimen_data_ex[0].specimen)
    
    save_json(os.path.join(OUTPUT, "erato_names.json"), metadata)
    np.savez(os.path.join(OUTPUT, "erato_dna_matrix.npz"), np.array(specimen_data))
    