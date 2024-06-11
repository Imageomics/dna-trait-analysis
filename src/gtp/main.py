import os

from tools import dna_to_vector

def get_lines(path):
    with open(path) as f:
        lines = [x.strip() for x in f.readlines()]
        lines = list(filter(lambda x: len(x) > 0, lines))
        return lines

def get_dna(srrs, root, type="raw_optix_scaffold"):
    dna_list = []
    for srr in srrs:
        path = os.path.join(root, srr + "_" + type + ".fa")
        dna = read_dna(path)
        dna_list.append(dna)
    return dna_list

def read_dna(x):
    with open(x) as f:
        lines = f.readlines()
        return lines[1]

if __name__ == "__main__":
    dataset_path = "/local/scratch/carlyn.1/dna/seqs"
    erato_srr_list = get_lines(os.path.join(dataset_path, "SRR_erato_5.txt"))
    melpo_srr_list = get_lines(os.path.join(dataset_path, "SRR_melpo_5.txt"))

    erato_dna = get_dna(erato_srr_list, os.path.join(dataset_path, "erato"))
    melpo_dna = get_dna(melpo_srr_list, os.path.join(dataset_path, "melpomene"))
    
    for dna in erato_dna + melpo_dna:
        print(dna[3000:3100])