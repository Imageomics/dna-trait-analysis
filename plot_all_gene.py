import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from tqdm import tqdm

from experiments import get_all_gene_experiments
from dataclasses import dataclass

RESULTS_DIR = Path("/home/carlyn.1/dna-trait-analysis/results")
EXP_DIR = RESULTS_DIR / "one_dim"

@dataclass
class GeneData:
    gene: str
    size: int

def load_att(species, wing, color):
    DATA_FOLDER = EXP_DIR / f"{species}_all_genes_{wing}_{color}"
    test_att = np.load(DATA_FOLDER / "att_points.npz")["test"]
    return test_att

for species in ["erato", "melpomene"]:
    for wing in ["forewings", "hindwings"]:
        exps = get_all_gene_experiments(species, wing, "color_1")
        gene_data = [GeneData(ex.gene.split("_")[-1], np.load(ex.gene_vcf_path)['arr_0'].shape[1]) for ex in exps]
        colors = ["purple", "green", "orange", "blue", "red"]

        fig, axs = plt.subplots(2, 2, figsize=(40, 20))
        FONT_SIZE = 18
        plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
        plt.rc('legend', fontsize=18)  # fontsize of the legend
        plt.rc('axes', labelsize='medium', titlesize='large')
        for c, color_type in enumerate(["color_1", "color_2", "color_3", "total"]):
            test_att = load_att(species, wing, color_type)
            ax = axs[c // 2, c % 2]

            ax.set_title(f"{species} | {exps[0].wing_side} | {color_type}", fontsize=32)
            ax.set_ylabel('Attributions', fontsize=20)
            ax.set_xlabel('VCF Position', fontsize=20)
            ax.tick_params(axis='both', labelsize=14)

            def normalize(att):
                att -= att.min()
                att /= att.max()
                return att

            test_att = np.abs(test_att)
            test_att = normalize(test_att)

            x_max = test_att.shape[0]
            #ax.set_xlim([0, x_max])
            X = np.arange(x_max)
            prev = 0
            handles = []
            for i, gd in tqdm(enumerate(gene_data), desc="Plotting Genes"):
                #print(prev, prev+gd.size, len(test_att))
                y = test_att[prev:prev+gd.size]
                alpha = 0.2 + y * 0.8
                s = y * 3
                ax.scatter(X[prev:prev+gd.size], y, alpha=alpha, s=s, color=colors[i], label=gd.gene)
                #ax.bar(X[prev:prev+gd.size], test_att[prev:prev+gd.size], align='center', alpha=0.1, s=0.1, color=colors[i], label=gd.gene)
                prev += gd.size
                handles.append(mpatches.Patch(color=colors[i], label=gd.gene))
            ax.set_ylim([0, 1.1])
            ax.legend(handles=handles)
            
        plt.savefig(EXP_DIR / f"{species}-{wing}.png")
        plt.tight_layout()
        plt.show()
        plt.close()