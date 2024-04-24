import numpy as np
from data_tools import get_chromo_info

with open("erato_sliding_window.txt", "r") as f:
    lines = f.readlines()

SPECIES = "erato"
vcf_file = f"/local/scratch/carlyn.1/dna/vcfs/{SPECIES}_optix_variants.tsv"

x = []
y1 = []
y2 = []
for line in lines:
    parts = line.split(" ")
    start = int(parts[1][1:-1])
    end = int(parts[2][:-1])
    val = float(parts[10][1:-1])
    test = float(parts[14][1:-2])

    x.append(f"{start}-{end}")
    y1.append(val)
    y2.append(test)


idx_sort = np.argsort(y1)
for idx in idx_sort[:5]:
    start, end = x[idx].split("-")
    c_s = get_chromo_info(vcf_file, int(start))[1]
    c_e = get_chromo_info(vcf_file, int(end))[1]
    print(f"Window: {x[idx]} | Chromo: {c_s}-{c_e} | Val Loss: {y1[idx]}")
#print(np.array(x)[idx_sort[:5]])
#print(np.array(chromo)[idx_sort[:5]])
#print(np.array(y1)[idx_sort[:5]])
    