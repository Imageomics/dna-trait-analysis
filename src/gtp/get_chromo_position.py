from data_tools import get_chromo_info
from argparse import ArgumentParser

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--pos", type=int, default=0)
    args = parser.parse_args()
    SPECIES = "erato"
    chrom_info = get_chromo_info(f"/local/scratch/carlyn.1/dna/vcfs/{SPECIES}_optix_variants.tsv", args.pos)
    out_str = f"Chromo: {chrom_info[0]} | Position: {chrom_info[1]} | Ref Allele: {chrom_info[2]} | Alt Alele: {chrom_info[3]}"
    print(out_str)