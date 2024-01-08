import os
from argparse import ArgumentParser

import numpy as np
from PIL import Image

import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--results_dir", type=str, default="results")
    args = parser.parse_args()
    return args

def combine_results(results_dir):
    genes = {}
    for root, dirs, paths in os.walk(results_dir):
        for p in paths:
            fname, ext = os.path.splitext(p)
            if ext != ".png": continue
            fname_parts = fname.split("_")
            if fname_parts[1] == "results": continue
            gene = fname_parts[1]
            if gene not in genes:
                genes[gene] = []
            genes[gene].append(os.path.join(root, p))

    
    for gene in genes:
        genes[gene].sort()
        imgs = [np.array(Image.open(p)) for p in genes[gene]]
        
        fig = plt.figure(figsize=(20, 20))
        grid = ImageGrid(fig, 111, nrows_ncols=(4, 3), axes_pad=0.1)
        for i, ax in enumerate(grid):
            ax.imshow(imgs[i])
            ax.set_axis_off()
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, f"{gene}_results.png"))
        plt.close()
    

if __name__ == "__main__":
    args = get_args()
    combine_results(args.results_dir)