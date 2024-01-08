import os
import json
import random
import time

from argparse import ArgumentParser

from tqdm import tqdm

from multiprocessing import Pool
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import numpy as np

import matplotlib.pyplot as plt

from evaluation import plot_attribution_graph

#from pytorch_grad_cam import GradCAM
#from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from models.net import ConvNet, SoyBeanNet, LargeNet
from data_tools import parse_patternize_csv
from create_curve_from_sliding_window import create_curve

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

class VCF_Dataset(Dataset):
    def __init__(self, data, norm_vals=None):
        super().__init__()
        self.data = data
        self.norm_vals = norm_vals

    def __getitem__(self, idx):
        name, in_data, out_data = self.data[idx]
        in_data = torch.tensor(in_data).unsqueeze(0).float()
        if self.norm_vals is not None:
            out_data = (out_data - self.norm_vals[0]) / self.norm_vals[1]
        out_data = torch.tensor(out_data).float()

        return name, in_data, out_data

    def __len__(self):
        return len(self.data)

def forward_step(model, batch, optimizer, args, is_train=True):
    name, data, pca = batch
    data = data.cuda()
    pca = pca[:, :args.out_dims].cuda()
    with torch.set_grad_enabled(is_train):
        if is_train:
            model.train()
        else:
            model.eval()
        out = model(data)
        loss = F.mse_loss(out, pca)
        rmse = torch.sqrt(loss).item()

        if is_train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
    return loss.item(), rmse

def train(args, tr_dloader, val_dloader, model, logger):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    best_model_weights = None
    best_err = 999999
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(args.epochs), desc="Training Batch of models", colour="green"):

        # Training
        total_rmse = 0
        for batch in tr_dloader:
            mse, rmse = forward_step(model, batch, optimizer, args, is_train=True)
            total_rmse += rmse
        
        avg_train_rmse = total_rmse / len(tr_dloader)
        
        # Validation
        total_rmse = 0
        for batch in val_dataloader:
            mse, rmse = forward_step(model, batch, None, args, is_train=False)
            total_rmse += rmse

        avg_val_rmse = total_rmse / len(val_dloader)

        if avg_val_rmse <= best_err:
            best_err = avg_val_rmse
            best_model_weights = model.state_dict()

        train_losses.append(avg_train_rmse)
        val_losses.append(avg_val_rmse)

        logger.log(f"Epoch {epoch+1}/{args.epochs}: Train RMSE: {avg_train_rmse} | Val RMSE: {avg_val_rmse}")
            
    model.load_state_dict(best_model_weights)
    model.eval()

    return train_losses, val_losses

def get_norm_vals(data):
    out_data = np.array([d[2] for d in data])

    mu = out_data.mean(axis=0)
    std = out_data.std(axis=0)

    return mu, std


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--out_dims", type=int, default=50)
    parser.add_argument("--insize", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--input_data", type=str, default="/local/scratch/carlyn.1/dna/vcfs/erato_dna_matrix.npz")
    parser.add_argument("--input_metadata", type=str, default="/local/scratch/carlyn.1/dna/vcfs/erato_names.json")
    parser.add_argument("--pca_loadings", type=str, default="/local/scratch/carlyn.1/dna/colors/erato_red_loadings.csv")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--verbose", action='store_true', default=False)

    return parser.parse_args()

class Logger:
    def __init__(self, args):
        self.args = args
        self.outdir = os.path.join(args.output_dir, args.exp_name)
        os.makedirs(self.outdir, exist_ok=True)

    def log(self, x):
        with open(os.path.join(self.outdir, "out.log"), 'a') as f:
            f.write(x + '\n')
            if self.args.verbose:
                print(x)

def plot_loss_curves(train_losses, val_losses, outdir):
    fig = plt.figure()
    ax = plt.gca()

    x = np.arange(len(train_losses))
    ax.plot(x, train_losses, label="Train Loss", color="red")
    ax.plot(x, val_losses, label="Val Loss", color="blue")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("RMSE")
    plt.title("Training Loss Curves")
    plt.legend()
    plt.tight_layout()

    plt.savefig(os.path.join(outdir, "loss_curves.png"))
    plt.close()

if __name__ == "__main__":
    args = get_args()
    logger = Logger(args)

    input_data = np.load(args.input_data)['arr_0']
    num_vcfs = input_data.shape[1]

    logger.log(f"Input size: {num_vcfs}")
    logger.log(f"Number of out dimensions used: {args.out_dims}")

    metadata = load_json(args.input_metadata)
    pca_data = parse_patternize_csv(args.pca_loadings)

    train_data = []
    for name, row in zip(metadata, input_data):
        if name+"_d" in pca_data:
            train_data.append([name, row, pca_data[name+"_d"]])

    random.seed(args.seed)
    random.shuffle(train_data)
    train_idx = int(len(train_data) * 0.8)
    val_idx = int(len(train_data) * 0.1)

    train_split = train_data[:train_idx]
    val_split = train_data[train_idx:train_idx+val_idx]
    test_split = train_data[train_idx+val_idx:]
    logger.log(f"Dataset sizes: train - {len(train_split)} | val - {len(val_split)} | test - {len(test_split)}")

    out_mu, out_std = get_norm_vals(train_split)

    #train_dataset = VCF_Dataset(train_split, norm_vals=(out_mu, out_std))
    #val_dataset = VCF_Dataset(val_split, norm_vals=(out_mu, out_std))
    #test_dataset = VCF_Dataset(test_split, norm_vals=(out_mu, out_std))
    train_dataset = VCF_Dataset(train_split)
    val_dataset = VCF_Dataset(val_split)
    test_dataset = VCF_Dataset(test_split)

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

    start_t = time.perf_counter()
    model = SoyBeanNet(window_size=num_vcfs, num_out_dims=args.out_dims, insize=args.insize).cuda()
    train_losses, val_losses = train(args, train_dataloader, val_dataloader, model=model, logger=logger)
    torch.save(model.state_dict(), os.path.join(logger.outdir, "model.pt"))
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    logger.log(f"Total training time: {total_duration:.2f}s")

    plot_loss_curves(train_losses, val_losses, logger.outdir)

    start_t = time.perf_counter()
    logger.log("Beginning attribution")
    plot_attribution_graph(model, train_dataloader, val_dataloader, test_dataloader, logger.outdir)
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    logger.log(f"Total attribution time: {total_duration:.2f}s")

    



