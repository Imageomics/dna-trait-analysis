import os
import json
import random
import time

from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt

from evaluation import plot_attribution_graph
from logger import Logger

from models.net import ConvNet, SoyBeanNet, LargeNet
from models.forward import forward_step
from data_tools import parse_patternize_csv, load_json, VCF_Dataset
from create_curve_from_sliding_window import create_curve

def train(args, tr_dloader, val_dloader, model, logger):
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)

    best_model_weights = None
    best_err = 999999
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(args.epochs), desc="Training", colour="green"):

        # Training
        total_rmse = 0
        for batch in tr_dloader:
            mse, rmse = forward_step(model, batch, optimizer, args.out_dims, is_train=True)
            total_rmse += rmse
        
        avg_train_rmse = total_rmse / len(tr_dloader)
        
        # Validation
        total_rmse = 0
        for batch in val_dataloader:
            mse, rmse = forward_step(model, batch, None, args.out_dims, is_train=False)
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

def load_data(args):
    input_data = np.load(args.input_data)['arr_0']

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

    return train_split, val_split, test_split

def log_data_splits(train_split, val_split, test_split, logger):
    logger.log(f"Dataset sizes: train - {len(train_split)} | val - {len(val_split)} | test - {len(test_split)}")
    for split_type, split in zip(["train", "val", "test"], [train_split, val_split, test_split]):
        out_str = ""
        for name, _, _ in split:
            out_str += name + ","
        out_str = out_str[:-1]
        with open(os.path.join(logger.outdir, f"{split_type}_split.txt"), 'w') as f:
            f.write(out_str)

if __name__ == "__main__":
    args = get_args()
    logger = Logger(args)

    train_split, val_split, test_split = load_data(args)
    log_data_splits(train_split, val_split, test_split, logger)

    num_vcfs = train_split[0][1].shape[0]

    logger.log(f"Input size: {num_vcfs}")
    logger.log(f"Number of out dimensions used: {args.out_dims}")

    out_mu, out_std = get_norm_vals(train_split)

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

    logger.log(f"Testing")
    rmses = test(train_dataloader, val_dataloader, test_dataloader, model, args.out_dims)
    logger.log(f"Train RMSE: {rmses[0]} | Val RMSE: {rmses[1]} | Test RMSE: {rmses[2]}")

    plot_loss_curves(train_losses, val_losses, logger.outdir)

    start_t = time.perf_counter()
    logger.log("Beginning attribution")
    plot_attribution_graph(model, train_dataloader, val_dataloader, test_dataloader, logger.outdir)
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    logger.log(f"Total attribution time: {total_duration:.2f}s")

    



