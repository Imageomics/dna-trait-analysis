"""
process_genome_data.py

Goal:   To take in .tsv files with VCF information and
        convert to a workable format.
"""
from tqdm import tqdm
from data_tools import VCF, parse_vcfs

def process_genome(path_to_tsv):
    """
    To take in .tsv files with VCF information and
    convert to a workable format (numpy array and metadata).

    Args:
        path_to_tsv (str): Path to .tsv file with VCF information
    """
    
    vcfs = parse_vcfs(path_to_tsv)
    
if __name__ == "__main__":
    test_path = "/local/scratch/carlyn.1/dna/vcfs/erato/Herato1001_wntA.tsv"
    process_genome(test_path)