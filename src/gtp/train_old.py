import os
import json
import copy
import random
import time

from argparse import ArgumentParser

from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

import numpy as np

import matplotlib.pyplot as plt

from evaluation import plot_attribution_graph, test, get_guided_gradcam_attr
from logger import Logger

from models.net import ConvNet, SoyBeanNet, LargeNet
from models.forward import forward_step
from models.scheduler import Scheduler
from data_tools import parse_patternize_csv, load_json, VCF_Dataset
from create_curve_from_sliding_window import create_curve
from experiments import get_experiment, get_all_gene_experiments

def get_optimizer(args, params):
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr)
    
    raise NotImplementedError(f"{args.optimizer} has not been implemented!")

def train(args, tr_dloader, val_dloader, model, logger):
    #optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    #optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = get_optimizer(args, model.parameters())
    scheduler = Scheduler(args, optimizer)

    best_model_weights = None
    best_err = 999999
    train_losses = []
    val_losses = []
    for epoch in tqdm(range(args.epochs), desc="Training", colour="green"):

        # Training
        total_rmse = 0
        for batch in tr_dloader:
            mse, rmse = forward_step(model, batch, optimizer, args.out_dims, out_start_idx=args.out_dims_start_idx, is_train=True)
            total_rmse += rmse
        scheduler.step()
        
        avg_train_rmse = total_rmse / len(tr_dloader)
        
        # Validation
        total_rmse = 0
        best_diff_e = 99999
        worst_diff_e = -99999
        for batch in val_dataloader:
            mse, rmse, best_diff, worst_diff = forward_step(model, batch, None, args.out_dims, is_train=False, return_diff=True)
            best_diff_e = min(best_diff_e, best_diff)
            worst_diff_e = max(worst_diff_e, worst_diff)
            total_rmse += rmse

        avg_val_rmse = total_rmse / len(val_dloader)

        if avg_val_rmse <= best_err:
            logger.log("Saving Model")
            best_err = avg_val_rmse
            best_model_weights = copy.deepcopy(model).state_dict()

        train_losses.append(avg_train_rmse)
        val_losses.append(avg_val_rmse)

        def rs(v: float) -> str:
            r = round(v, 4)
            return f"{r:.4f}"
        out_str = f"Epoch {epoch+1}/{args.epochs}: Train RMSE: {rs(avg_train_rmse)} | Val RMSE: {rs(avg_val_rmse)}"
        out_str += f" | Best Diff: {rs(best_diff_e)} | Worst Diff: {rs(worst_diff_e)}"
        logger.log(out_str)
            
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
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--drop_out_prob", type=float, default=0.75)
    parser.add_argument("--out_dims", type=int, default=10)
    parser.add_argument("--out_dims_start_idx", type=int, default=0)
    parser.add_argument("--insize", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--species", type=str, default="erato")
    parser.add_argument("--gene", type=str, default="optix")
    parser.add_argument("--color", type=str, default="color_3")
    parser.add_argument("--wing", type=str, default="forewings")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument("--output_dir", type=str, default="/home/carlyn.1/dna-trait-analysis/results")
    parser.add_argument("--verbose", action='store_true', default=False)
    parser.add_argument("--all_genes", action='store_true', default=False)
    parser.add_argument("--is_large", action='store_true', default=False)
    parser.add_argument("--scheduler", type=str, default="none")
    parser.add_argument("--optimizer", type=str, default="adam")

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

def load_data(args, experiments):
    pca_data = parse_patternize_csv(experiments[0].pca_loading_path)
    for i, experiment in enumerate(experiments):
        if i == 0:
            input_data = np.load(experiment.gene_vcf_path)['arr_0']
            metadata = load_json(experiment.metadata_path)
        else:
            input_data = np.hstack((input_data, np.load(experiment.gene_vcf_path)['arr_0']))
            new_metadata = load_json(experiment.metadata_path)
            pca_data = parse_patternize_csv(experiment.pca_loading_path)
            for j, m in enumerate(metadata):
                assert m == new_metadata[j], f"Metadata does not match: {m} != {new_metadata[j]}"
            

    train_data = []
    print(f"Length of input data: {len(input_data)}")
    for name, row in zip(metadata, input_data):
        if name+"_d" in pca_data:
            train_data.append([name, row, pca_data[name+"_d"]])
    print(f"Length of train data: {len(train_data)}")

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

def setup():
    args = get_args()
    if args.all_genes:
        experiments = get_all_gene_experiments(args.species, args.wing, args.color)
        exp_name_d = experiments[0].get_experiment_name()
        print(exp_name_d)
        parts = exp_name_d.split("_")
        parts[1] = "all"
        parts[2] = "genes"
        exp_name_d = "_".join(parts)
        gene_str = "all_genes"
    else:
        experiments = [get_experiment(args.species, args.gene, args.wing, args.color, is_large=args.is_large)]
        exp_name_d = experiments[0].get_experiment_name()
        gene_str = experiments[0].gene
    
    logger = Logger(args, exp_name=exp_name_d)
    exp = experiments[0]
    
    logger.log(f"""
    Species: {exp.species}
    Gene: {gene_str}
    Wing: {exp.wing_side}
    Color: {exp.pca_type}
    """)

    data = load_data(args, experiments)
    
    return args, experiments, logger, data

if __name__ == "__main__":
    args, experiments, logger, data = setup()

    train_split, val_split, test_split = data
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
    model = SoyBeanNet(window_size=num_vcfs, num_out_dims=args.out_dims, insize=args.insize, hidden_dim=args.hidden_dim, drop_out_prob=args.drop_out_prob).cuda()
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
    plot_attribution_graph(model, train_dataloader, val_dataloader, test_dataloader, logger.outdir, ignore_train=True, mode="cam", ignore_plot=True)
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    logger.log(f"Total attribution time: {total_duration:.2f}s")

    



