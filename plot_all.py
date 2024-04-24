import os
from argparse import ArgumentParser

import numpy as np

import matplotlib.pyplot as plt

from experiments import get_all_experiments, get_all_genes, get_all_colors, get_all_wing_sides, get_all_species
from data_tools import get_chromo_info

PLOT_COLORS = {
    "color_1" : "grey",
    "color_2" : "brown",
    "color_3" : "orange",
    "total" : "green"
}

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--plot_dir", type=str, default="plot_results")
    parser.add_argument("--split", type=str, default="test")
    parser.add_argument("--quick_run", action="store_true", default=False)

    return parser.parse_args()

def plot(group_data, plot_dir="plot_results", run_type="test"):
    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot()
    exp_ex = group_data["color_1"][1]
    fig.suptitle(f"{exp_ex.gene} | {exp_ex.wing_side}", fontsize=32)
    plt.ylabel('Attributions', fontsize=20)
    plt.xlabel('VCF Position', fontsize=20)

    FONT_SIZE = 18
    plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
    plt.rc('legend', fontsize=FONT_SIZE)  # fontsize of the legend
    plt.rc('axes', labelsize='medium', titlesize='large')

    ax.tick_params(axis='both', labelsize=14)

    ax.set_ylim([0, 1.1])
    ax.set_xlim([0, len(group_data["color_1"][0])])
    for pca_type in group_data:
        pts, exp, arrow_data = group_data[pca_type]
        pca_plot_color = PLOT_COLORS[pca_type]
        print(f"Plotting: {pca_type}")
        y = pts
        ax.bar(np.arange(len(pts)), y, align='center', alpha=0.65, color=pca_plot_color, label=pca_type)
        print(f"End Plotting: {pca_type}")

        if arrow_data:
            ar_x, ar_y = arrow_data[:2]
            chromo_pos = arrow_data[-1]
            ax.annotate(chromo_pos,
                        xy=(ar_x, ar_y),
                        xycoords="data",
                        xytext=(ar_x, ar_y + 0.05),
                        arrowprops=dict(facecolor=pca_plot_color, shrink=0.05),
                        horizontalalignment='center',
                        verticalalignment='bottom')

    #ax.autoscale_view()
    ax.legend()
    plt.tight_layout()
    fname = f"{run_type}_attribution_scaled_{exp_ex.gene}_{exp_ex.wing_side}"
    plt.savefig(os.path.join(plot_dir, f"{fname}.png"))
    plt.close()

def get_gene_root(full_gene):
    return full_gene.split("_")[-1]

if __name__ == "__main__":
    args = get_args()

    experiments = get_all_experiments()

    # Collect attribution points
    plot_points = []
    min_v = 9999
    max_v = -9999
    for i, exp in enumerate(experiments):
        results_dir = os.path.join(args.output_dir, exp.get_experiment_name())
        pt_path = os.path.join(results_dir, f"att_points.npz")
        test_pts = np.abs(np.load(pt_path)["test"]) # ABS value to get signal
        if args.quick_run:
            test_pts = test_pts[:2000]
        min_v = min(min_v, test_pts.min())
        max_v = max(max_v, test_pts.max())
        plot_points.append(test_pts)

    # Scale points
    # Get annotation data
    arrow_data = []
    print(f"Max value: {max_v} | Min value: {min_v}")
    for i, _ in enumerate(plot_points):
        plot_points[i] -= min_v
        plot_points[i] /= (max_v-min_v)

        max_i = np.argmax(plot_points[i])
        plot_max_v = plot_points[i][max_i]
        c_s = get_chromo_info(experiments[i].gene_vcf_tsv_path, max_i)[1]

        arrow_data.append([max_i, plot_max_v, str(c_s)])

    unique_genes = set()
    for genes in get_all_genes().values():
        for g in genes:
            unique_genes.add(get_gene_root(g))

    group_data = {
        g: {
            x: {
                y: {} for y in get_all_wing_sides()
            } for x in get_all_species()
        } for g in unique_genes
    }

    for pts, exp, ar_data in zip(plot_points, experiments, arrow_data):
        gene = get_gene_root(exp.gene)
        group_data[gene][exp.species][exp.wing_side][exp.pca_type] = [
            pts, exp, ar_data
        ]

    for gene in unique_genes:
        for species in get_all_species():
            for wing_side in get_all_wing_sides():
                print(f"Plotting: {gene} | {species} | {wing_side}")
                plot(group_data[gene][species][wing_side],
                    plot_dir=args.plot_dir,
                    run_type=args.split)