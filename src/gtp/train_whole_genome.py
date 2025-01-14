import copy
import os
import time
from argparse import ArgumentParser
from concurrent.futures import ThreadPoolExecutor, as_completed

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from gtp.dataloading.datasets import GTP_Dataset
from gtp.dataloading.tools import (
    load_chromosome_data,
    split_data_by_file,
)
from gtp.evaluation import plot_attribution_graph, test
from gtp.logger import Logger
from gtp.models.net import SoyBeanNet
from gtp.models.scheduler import Scheduler
from gtp.tools.calculation import calc_pvalue_linear, filter_topk_snps
from gtp.tools.timing import profile_exe_time
from gtp.trainers.trackers import DNATrainingTracker
from gtp.trainers.training_loops import BasicTrainingLoop


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
    optimizer = get_optimizer(args, model.parameters())
    scheduler = Scheduler(args, optimizer)

    best_model_weights = None
    best_err = 999999
    best_pearson = 0
    train_losses = []
    val_losses = []
    val_pearsons = []
    training_loop = BasicTrainingLoop(options=None)
    training_tracker = DNATrainingTracker()
    for epoch in tqdm(range(args.epochs), desc="Training", colour="green"):
        # Training
        model.train()
        training_loop.train(
            tr_dloader,
            model=lambda batch: model(batch[0].cuda()),
            loss_fn=lambda output, batch: F.mse_loss(batch[1].cuda(), output),
            optimizer=optimizer,
            tracker=training_tracker,
        )
        scheduler.step()

        total_rmse = sum(training_tracker.data_storage["training_rmse"])
        avg_train_rmse = total_rmse / len(tr_dloader)

        # Validation
        total_rmse = 0
        best_diff_e = 99999
        worst_diff_e = -99999
        model.eval()
        training_loop.test(
            val_dloader,
            model=lambda batch: model(batch[0].cuda()),
            loss_fn=lambda output, batch: F.mse_loss(batch[1].cuda(), output),
            tracker=training_tracker,
        )

        total_rmse = sum(training_tracker.data_storage["testing_rmse"])
        avg_val_rmse = total_rmse / len(val_dloader)

        training_tracker.reset_data_storage()

        # TODO: calc best and worst error difference (between pca and prediction)
        best_diff_e = 0  # min(best_diff_e, best_diff)
        worst_diff_e = 0  # max(worst_diff_e, worst_diff)

        pearson_stat, pval = calc_pearson_correlation(model, val_dloader)

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
    parser.add_argument(
        "--top_k_chromosome_training", action="store_true", default=False
    )
    parser.add_argument("--top_k_chromosome_training_path", type=str, default=False)
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


@profile_exe_time(verbose=False)
def load_data(args):
    camids_aligned, genotype_data_aligned, phenotype_data_aligned = (
        load_chromosome_data(
            args.genome_folder,
            args.phenotype_folder,
            args.species,
            args.wing,
            args.color,
            args.chromosome,
        )
    )

    phenotype_data_aligned = phenotype_data_aligned[
        :, args.out_dims_start_idx : args.out_dims_start_idx + args.out_dims
    ]

    train_split, val_split, test_split = split_data_by_file(
        genotype_data_aligned,
        phenotype_data_aligned,
        camids_aligned,
        args.split_data_folder,
        args.species,
    )

    return train_split, val_split, test_split


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


@profile_exe_time(verbose=False)
def setup():
    args = get_args()

    exp_name = f"{args.species}_{args.wing}_{args.color}_chromosome_{args.chromosome}"
    if args.top_k_chromosome_training:
        exp_name = f"{args.species}_{args.wing}_{args.color}_top_k_snps"

    logger = Logger(args, exp_name=exp_name)

    logger.log(f"""
    Species: {args.species}
    Chromosome: {args.chromosome}
    Wing: {args.wing}
    Color: {args.color}
    """)

    if args.top_k_chromosome_training:

        def load_one(args, chromosome, snp_idx, idx):
            cur_args = copy.deepcopy(args)
            cur_args.chromosome = chromosome
            train_data, val_data, test_data = load_data(cur_args)
            snp_idx = np.sort(snp_idx).astype(np.int64)
            train_data[0] = train_data[0][:, snp_idx]
            val_data[0] = val_data[0][:, snp_idx]
            test_data[0] = test_data[0][:, snp_idx]
            return (
                idx,
                train_data,
                val_data,
                test_data,
            )

        futures = []
        pool = ThreadPoolExecutor(21)
        final_train_data = None
        final_val_data = None
        final_test_data = None
        data = np.load(args.top_k_chromosome_training_path, allow_pickle=True)
        test_snp_selections = data.item()["test"]
        for idx, snp_idx in tqdm(
            enumerate(test_snp_selections), desc="loading top snps from chromosome"
        ):
            future = pool.submit(load_one, args, idx + 1, snp_idx, idx)
            futures.append(future)

        total = 0
        all_data = []
        for future in as_completed(futures):
            proc_idx, train_data, val_data, test_data = future.result()
            all_data.append([proc_idx, train_data, val_data, test_data])
            total += 1
            print(f"Completed loading on chromosome: {proc_idx+1}: ({total}/21)")

        for proc_idx, train_data, val_data, test_data in sorted(
            all_data, key=lambda x: int(x[0])
        ):
            if final_train_data is None:
                final_train_data = train_data
                final_val_data = val_data
                final_test_data = test_data
            else:
                final_train_data[0] = np.concatenate(
                    (final_train_data[0], train_data[0]), axis=1
                )
                final_val_data[0] = np.concatenate(
                    (final_val_data[0], val_data[0]), axis=1
                )
                final_test_data[0] = np.concatenate(
                    (final_test_data[0], test_data[0]), axis=1
                )
                assert (train_data[1] == final_train_data[1]).all()
                assert (val_data[1] == final_val_data[1]).all()
                assert (test_data[1] == final_test_data[1]).all()

        return args, logger, final_train_data, final_val_data, final_test_data
    else:
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


def plot_pvalue_filtering(test_pts, test_dataset, logger, prefix="", k=200):
    top_k_idx = filter_topk_snps(test_pts, k=k)

    pvals = calc_pvalue_linear(
        np.take(test_dataset.genotype_data, indices=top_k_idx, axis=1),
        test_dataset.phenotype_data[:, 0],
    )
    all_pvals = np.ones(len(test_pts))
    all_pvals[top_k_idx] = pvals

    plt.figure(figsize=(20, 10))
    ax = plt.subplot()
    ax.set_title("SNP values")
    ax.set_ylabel("-log(p_value)")
    FONT_SIZE = 16
    plt.rc("font", size=FONT_SIZE)  # fontsize of the text sizes
    plt.rc("axes", titlesize=FONT_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=FONT_SIZE)  # fontsize of the x and y labels
    plt.rc("legend", fontsize=FONT_SIZE - 4)  # fontsize of the legend

    print("Plotting")
    y = -np.log(all_pvals)
    ax.scatter(
        np.arange(len(test_pts))[top_k_idx],
        y[top_k_idx],
        alpha=0.8,
        color="#eb5e7c",
    )
    ax.axhline(y=-np.log(1e-8), color="red")
    ax.axhline(y=-np.log(1e-5), color="orange")
    ax.axhline(y=-np.log(0.05 / len(all_pvals)), color="green")
    ax.axhline(y=-np.log(0.05 / k), color="blue")
    print("End Plotting")
    # ax.autoscale_view()
    plt.tight_layout()
    if prefix != "":
        prefix = f"{prefix}_"
    plt.savefig(os.path.join(logger.outdir, f"{prefix}topk_threshold_pvalues.png"))
    plt.close()


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
    for att_method in ["cam", "lrp"]:
        tr_pts, val_pts, test_pts = plot_attribution_graph(
            model,
            train_dataloader,
            val_dataloader,
            test_dataloader,
            logger.outdir,
            ignore_train=True,
            mode=att_method,
            ignore_plot=False,
            use_new=True,
        )

        k = 2000
        if args.top_k_chromosome_training:
            k = num_vcfs
        plot_pvalue_filtering(test_pts, test_dataset, logger, prefix=att_method, k=-1)

    end_t = time.perf_counter()
    total_duration = end_t - start_t
    logger.log(f"Total attribution time: {total_duration:.2f}s")
