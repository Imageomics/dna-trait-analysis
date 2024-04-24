import os
import json
import copy
import random
import time

from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, Dataset


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from logger import Logger

from models.net import ConvNet, SoyBeanNet, LargeNet
from models.forward import forward_step
from models.scheduler import Scheduler
from data_tools import parse_patternize_csv, load_json
from create_curve_from_sliding_window import create_curve
from experiments import get_experiment, get_all_gene_experiments, get_all_genes

from captum.attr import GuidedGradCam
from dataclasses import dataclass

class VCF_Dataset(Dataset):
    def __init__(self, data, norm_vals=None):
        super().__init__()
        self.data = data
        self.norm_vals = norm_vals

    def __getitem__(self, idx):
        in_data = self.data[idx]
        in_data = torch.tensor(in_data).unsqueeze(0).float()
        return in_data

    def __len__(self):
        return len(self.data)

def compile_attribution(attr):
    attr, _ = attr.max(-1)
    attr = attr[:, 0] # Only has 1 channel, just extract it
    attr = attr.abs().sum(0) # Sum across batch...Should we be summing here???
    return attr.detach().cpu().numpy()

def get_guided_gradcam_attr(m, dloader):
    att_model = GuidedGradCam(m, m.block)
    attr_total = None
    for i, batch in enumerate(dloader):
        m.zero_grad()
        data = batch
        attr = att_model.attribute(data.cuda(), 0)
        attr = compile_attribution(attr)
        if attr_total is None:
            attr_total = attr
        else:
            attr_total += attr

    #attr_total = attr_total / np.linalg.norm(attr_total, ord=1) # Normalize
    return attr_total

ROOT_PATH = Path("/home/carlyn.1/dna-trait-analysis/results/feb19")
INPUT_PATH_ROOT = Path("/local/scratch/carlyn.1/dna/vcfs")
genes = get_all_genes()["erato"]
for i, gene in enumerate(genes):
    if i == 0:
        input_data = np.load(INPUT_PATH_ROOT / (gene + "_vcfs.npz"))['arr_0']
        meta_data = load_json(INPUT_PATH_ROOT / (gene + "_names.json"))
    else:
        input_data = np.hstack((input_data, np.load(INPUT_PATH_ROOT / (gene + "_vcfs.npz"))['arr_0']))
        meta_data = load_json(INPUT_PATH_ROOT / (gene + "_names.json"))

dirs = [f"all_genes_forewings_{color_type}" for color_type in ['color_1', 'color_2', 'color_3', 'total']]

for dir in dirs:
    data_txt = ROOT_PATH / Path(dir) / "test_split.txt"
    with open(data_txt, 'r') as f:
        ids = []
        for l in f.readlines():
            l = l.strip()
            ids.extend(l.split(","))

    cur_input_data = None
    for i, (name, row) in enumerate(zip(meta_data, input_data)):        
        if name in ids:
            if cur_input_data is None:
                cur_input_data = row[np.newaxis, :, :]
            else:
                cur_input_data = np.vstack((cur_input_data, row[np.newaxis, :, :]))
                

    test_dataset = VCF_Dataset(cur_input_data)
    test_dataloader = DataLoader(test_dataset, batch_size=512, num_workers=8, shuffle=False)
    
    MODEL_PATH = Path("/home/carlyn.1/dna-trait-analysis/results/feb19/") / dir / "model.pt"

    model = SoyBeanNet(window_size=340202, num_out_dims=10, insize=3, hidden_dim=10).cuda()
    weights = torch.load(MODEL_PATH)
    model.load_state_dict(weights)

    model.eval()
    test_att = get_guided_gradcam_attr(model, test_dataloader)

    @dataclass
    class GeneData:
        gene: str
        size: int

    exps = get_all_gene_experiments("erato", "forewings", "color_1")
    gene_data = [GeneData(ex.gene.split("_")[-1], np.load(ex.gene_vcf_path)['arr_0'].shape[1]) for ex in exps]
    gene_data

    colors = ["purple", "green", "orange", "blue", "red"]

    fig = plt.figure(figsize=(20, 10))
    ax = plt.subplot()
    fig.suptitle(f"All Genes | {exps[0].wing_side}", fontsize=32)
    plt.ylabel('Attributions', fontsize=20)
    plt.xlabel('VCF Position', fontsize=20)

    FONT_SIZE = 18
    plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
    plt.rc('legend', fontsize=FONT_SIZE)  # fontsize of the legend
    plt.rc('axes', labelsize='medium', titlesize='large')

    ax.tick_params(axis='both', labelsize=14)

    #ax.set_ylim([0, 1.1])
    test_att = np.abs(test_att)
    x_max = test_att.shape[0]
    #ax.set_xlim([0, x_max])
    X = np.arange(x_max)
    prev = 0
    for i, gd in tqdm(enumerate(gene_data), desc="Plotting Genes"):
        ax.bar(X[prev:prev+gd.size], test_att[prev:prev+gd.size], align='center', alpha=0.65, color=colors[i], label=gd.gene)
        prev += gd.size
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close()
    plt.savefig(f"{dir}.png")
