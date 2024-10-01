import copy
import os
import time
from argparse import ArgumentParser

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from gtp.dataloading.datasets import GTP_Dataset
from gtp.dataloading.tools import collect_chromosome
from gtp.evaluation import plot_attribution_graph, test
from gtp.logger import Logger
from gtp.models.net import SoyBeanNet
from gtp.models.scheduler import Scheduler


def get_optimizer(args, params):
    if args.optimizer == "sgd":
        return torch.optim.SGD(params, lr=args.lr)
    if args.optimizer == "adam":
        return torch.optim.Adam(params, lr=args.lr)

    raise NotImplementedError(f"{args.optimizer} has not been implemented!")


def calc_pearson_correlation(model, dloader):
    model.eval()
    actual = []
    predicted = []
    for i, batch in enumerate(dloader):
        model.zero_grad()
        data, pca = batch
        out = model(data.cuda())
        actual.extend(pca[:, 0].detach().cpu().numpy().tolist())
        predicted.extend(out[:, 0].detach().cpu().numpy().tolist())
    pr = pearsonr(predicted, actual)
    return pr.statistic, pr.pvalue


def forward_step(model, batch, optimizer, is_train=True, return_diff=False):
    data, pca = batch
    data = data.cuda()
    pca = pca.cuda()
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

        if return_diff:
            with torch.no_grad():
                diff = (out - pca).abs()[:, 0]
                best_diff = min(diff).item()
                worst_diff = max(diff).item()
                return loss.item(), rmse, best_diff, worst_diff
    return loss.item(), rmse


def train(args, tr_dloader, val_dloader, model, logger):
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr)
    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = get_optimizer(args, model.parameters())
    scheduler = Scheduler(args, optimizer)

    best_model_weights = None
    best_err = 999999
    best_pearson = 0
    train_losses = []
    val_losses = []
    val_pearsons = []
    for epoch in tqdm(range(args.epochs), desc="Training", colour="green"):
        # Training
        total_rmse = 0
        for batch in tr_dloader:
            mse, rmse = forward_step(model, batch, optimizer, is_train=True)
            total_rmse += rmse
        scheduler.step()

        avg_train_rmse = total_rmse / len(tr_dloader)

        # Validation
        total_rmse = 0
        best_diff_e = 99999
        worst_diff_e = -99999
        for batch in val_dataloader:
            mse, rmse, best_diff, worst_diff = forward_step(
                model, batch, None, is_train=False, return_diff=True
            )
            best_diff_e = min(best_diff_e, best_diff)
            worst_diff_e = max(worst_diff_e, worst_diff)
            total_rmse += rmse

        avg_val_rmse = total_rmse / len(val_dloader)

        pearson_stat, pval = calc_pearson_correlation(model, val_dataloader)

        if args.save_stat == "loss" and avg_val_rmse <= best_err:
            logger.log("Saving Model")
            best_err = avg_val_rmse
            best_model_weights = copy.deepcopy(model).state_dict()
        elif args.save_stat == "pearson" and pearson_stat >= best_pearson:
            logger.log("Saving Model")
            best_pearson = pearson_stat
            best_model_weights = copy.deepcopy(model).state_dict()

        train_losses.append(avg_train_rmse)
        val_losses.append(avg_val_rmse)
        val_pearsons.append(pearson_stat)

        def rs(v: float) -> str:
            r = round(v, 4)
            return f"{r:.4f}"

        out_str = f"Epoch {epoch+1}/{args.epochs}: Train RMSE: {rs(avg_train_rmse)} | Val RMSE: {rs(avg_val_rmse)} | Val Pearson: {pearson_stat}"
        out_str += f" | Best Diff: {rs(best_diff_e)} | Worst Diff: {rs(worst_diff_e)}"
        logger.log(out_str)

    model.load_state_dict(best_model_weights)
    model.eval()

    return train_losses, val_losses, val_pearsons


def get_norm_vals(data):
    out_data = np.array([d[2] for d in data])

    mu = out_data.mean(axis=0)
    std = out_data.std(axis=0)

    return mu, std


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--drop_out_prob", type=float, default=0.75)
    parser.add_argument("--out_dims", type=int, default=1)
    parser.add_argument("--out_dims_start_idx", type=int, default=0)
    parser.add_argument("--insize", type=int, default=3)
    parser.add_argument("--hidden_dim", type=int, default=10)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--species", type=str, default="erato")
    parser.add_argument(
        "--genome_folder",
        type=str,
        default="/local/scratch/carlyn.1/dna/vcfs/processed/genome",
    )
    parser.add_argument(
        "--phenotype_folder",
        type=str,
        default="/local/scratch/carlyn.1/dna/colors/processed",
    )
    parser.add_argument(
        "--split_data_folder",
        type=str,
        default="/home/carlyn.1/dna-trait-analysis/data",
    )
    parser.add_argument("--chromosome", type=int, default=1)
    parser.add_argument("--color", type=str, default="total")
    parser.add_argument("--wing", type=str, default="forewings")
    parser.add_argument("--exp_name", type=str, default="debug")
    parser.add_argument(
        "--output_dir", type=str, default="/local/scratch/carlyn.1/dna/results"
    )
    parser.add_argument("--verbose", action="store_true", default=False)
    parser.add_argument("--scheduler", type=str, default="none")
    parser.add_argument("--optimizer", type=str, default="adam")
    parser.add_argument("--save_stat", type=str, default="pearson")

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
    """
    NOTE: There are missing camids from either pheno or geno type data
    Seems to consistently be 1 missing from melpomene genotype
    and 4 missing from erato phenotype
    """
    pca_path = os.path.join(
        args.phenotype_folder, f"{args.species}_{args.wing}_{args.color}", "data.csv"
    )
    pca_df = pd.read_csv(pca_path)
    pca_camids = pca_df.camid.to_numpy()

    sort_idx = np.argsort(pca_camids)
    pca_camids_sorted = pca_camids[sort_idx]

    pca_data = pca_df.iloc[
        :, (1 + args.out_dims_start_idx) : (1 + args.out_dims_start_idx + args.out_dims)
    ].to_numpy()
    pca_data = pca_data[sort_idx]

    sorted_camids, compiled_data = collect_chromosome(
        args.genome_folder, args.species, args.chromosome
    )
    conflict_camid_a = list(
        set.difference(set(pca_camids_sorted.tolist()), set(sorted_camids.tolist()))
    )
    conflict_camid_b = list(
        set.difference(set(sorted_camids.tolist()), set(pca_camids_sorted.tolist()))
    )
    conflict_camids = conflict_camid_a + conflict_camid_b

    idx = np.isin(sorted_camids, conflict_camids)
    sorted_camids = sorted_camids[~idx]
    compiled_data = compiled_data[~idx]

    idx = np.isin(pca_camids_sorted, conflict_camids)
    pca_camids_sorted = pca_camids_sorted[~idx]
    pca_data = pca_data[~idx]

    assert (pca_camids_sorted == sorted_camids).all(), "Invalid alignment"

    train_cams = np.load(
        os.path.join(args.split_data_folder, f"{args.species}_train.npy")
    )
    val_cams = np.load(os.path.join(args.split_data_folder, f"{args.species}_val.npy"))
    test_cams = np.load(
        os.path.join(args.split_data_folder, f"{args.species}_test.npy")
    )

    train_idx = np.isin(sorted_camids, train_cams)
    train_pca_data = pca_data[train_idx]
    train_geno_data = compiled_data[train_idx]

    val_idx = np.isin(sorted_camids, val_cams)
    val_pca_data = pca_data[val_idx]
    val_geno_data = compiled_data[val_idx]

    test_idx = np.isin(sorted_camids, test_cams)
    test_pca_data = pca_data[test_idx]
    test_geno_data = compiled_data[test_idx]

    return (
        [train_geno_data, train_pca_data],
        [val_geno_data, val_pca_data],
        [test_geno_data, test_pca_data],
    )


def log_data_splits(train_split, val_split, test_split, logger):
    logger.log(
        f"Dataset sizes: train - {len(train_split)} | val - {len(val_split)} | test - {len(test_split)}"
    )
    for split_type, split in zip(
        ["train", "val", "test"], [train_split, val_split, test_split]
    ):
        out_str = ""
        for name, _, _ in split:
            out_str += name + ","
        out_str = out_str[:-1]
        with open(os.path.join(logger.outdir, f"{split_type}_split.txt"), "w") as f:
            f.write(out_str)


def setup():
    args = get_args()

    exp_name = f"{args.species}_{args.wing}_{args.color}_chromosome_{args.chromosome}"

    logger = Logger(args, exp_name=exp_name)

    logger.log(f"""
    Species: {args.species}
    Chromosome: {args.chromosome}
    Wing: {args.wing}
    Color: {args.color}
    """)

    train_data, val_data, test_data = load_data(args)

    return args, logger, train_data, val_data, test_data


def test(tr_dloader, val_dloader, test_dloader, model):
    rmses = []
    for dl in [tr_dloader, val_dloader, test_dloader]:
        total_rmse = 0
        for batch in dl:
            mse, rmse = forward_step(model, batch, None, is_train=False)
            total_rmse += rmse

        avg_rmse = total_rmse / len(dl)
        rmses.append(avg_rmse)

    return rmses


if __name__ == "__main__":
    args, logger, train_data, val_data, test_data = setup()

    num_vcfs = train_data[0].shape[1]
    logger.log(f"Input size: {num_vcfs}")
    logger.log(f"Number of out dimensions used: {args.out_dims}")

    train_dataset = GTP_Dataset(*train_data)
    val_dataset = GTP_Dataset(*val_data)
    test_dataset = GTP_Dataset(*test_data)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False
    )

    start_t = time.perf_counter()
    model = SoyBeanNet(
        window_size=num_vcfs,
        num_out_dims=args.out_dims,
        insize=args.insize,
        hidden_dim=args.hidden_dim,
        drop_out_prob=args.drop_out_prob,
    ).cuda()
    train_losses, val_losses, val_pearsons = train(
        args, train_dataloader, val_dataloader, model=model, logger=logger
    )
    torch.save(model.state_dict(), os.path.join(logger.outdir, "model.pt"))
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    logger.log(f"Total training time: {total_duration:.2f}s")

    logger.log("Testing")
    rmses = test(train_dataloader, val_dataloader, test_dataloader, model)
    logger.log(f"Train RMSE: {rmses[0]} | Val RMSE: {rmses[1]} | Test RMSE: {rmses[2]}")

    plot_loss_curves(train_losses, val_losses, logger.outdir)

    start_t = time.perf_counter()
    logger.log("Beginning attribution")
    plot_attribution_graph(
        model,
        train_dataloader,
        val_dataloader,
        test_dataloader,
        logger.outdir,
        ignore_train=True,
        mode="cam",
        ignore_plot=True,
        use_new=True,
    )
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    logger.log(f"Total attribution time: {total_duration:.2f}s")
