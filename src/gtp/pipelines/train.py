import copy
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr
from torch.utils.data import DataLoader
from tqdm import tqdm

from gtp.configs.loaders import load_configs
from gtp.configs.project import GenotypeToPhenotypeConfigs
from gtp.dataloading.data_collectors import load_training_data
from gtp.dataloading.datasets import GTP_Dataset
from gtp.dataloading.path_collectors import get_results_training_output_directory
from gtp.evaluation import test
from gtp.models.net import SoyBeanNet, DeepNet
from gtp.models.scheduler import Scheduler
from gtp.options.training import TrainingOptions
from gtp.tools.calculation import calc_pvalue_linear, filter_topk_snps
from gtp.tools.logging import ExperimentLogger
from gtp.tools.simple import create_exp_info_text
from gtp.trainers.trackers import DNATrainingTracker
from gtp.trainers.training_loops import BasicTrainingLoop


def get_optimizer(optimizer: str, lr: float, params: torch.nn.parameter.Parameter):
    if optimizer == "sgd":
        return torch.optim.SGD(params, lr=lr)
    if optimizer == "adam":
        return torch.optim.Adam(params, lr=lr)

    raise NotImplementedError(f"{optimizer} has not been implemented!")


def calc_pearson_correlation(model, dloader):
    """We only care about the correlation between the observed prediction and the actual prediction.
       The pvalue is irrelevant here since we have no hypothesis. When multiple dimensions are given for
       y, then the average between all correlation coefficients is calcualted and returned.

    Args:
        model (_type_): _description_
        dloader (_type_): _description_

    Returns:
        float: mean pearson correlation coefficient between all output predicted and actuals
    """
    model.eval()
    actual = []
    predicted = []
    for i, batch in enumerate(dloader):
        with torch.no_grad():
            data, pca = batch
            if len(actual) == 0:
                actual.extend([[] for _ in range(pca.shape[1])])
                predicted.extend([[] for _ in range(pca.shape[1])])
            out = model(data.cuda())
            for d in range(pca.shape[1]):
                actual[d].extend(pca[:, d].detach().cpu().numpy().tolist())
                predicted[d].extend(out[:, d].detach().cpu().numpy().tolist())

    pearson_stats = []
    for act, pred in zip(actual, predicted):
        pr = pearsonr(pred, act)
        pearson_stats.append(pr.statistic)
    return np.array(pearson_stats).mean()


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

    plt.savefig(Path(outdir, "loss_curves.png"))
    plt.close()


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
    plt.savefig(Path(logger.outdir, f"{prefix}topk_threshold_pvalues.png"))
    plt.close()


def train_model(
    options: TrainingOptions,
    tr_dloader: DataLoader,
    val_dloader: DataLoader,
    model: SoyBeanNet,
    logger: ExperimentLogger,
):
    optimizer = get_optimizer(options.optimizer, options.lr, model.parameters())
    scheduler = Scheduler(options.scheduler, optimizer)

    best_model_weights = None
    best_err = 999999
    best_pearson = -2
    train_losses = []
    val_losses = []
    val_pearsons = []
    training_loop = BasicTrainingLoop(options=None)
    training_tracker = DNATrainingTracker()
    for epoch in tqdm(
        range(options.epochs),
        desc="Training",
        colour="green",
        disable=not options.verbose,
    ):
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

        pearson_stat = calc_pearson_correlation(model, val_dloader)

        if options.save_stat == "loss" and avg_val_rmse <= best_err:
            logger.log("Saving Model")
            best_err = avg_val_rmse
            best_model_weights = copy.deepcopy(model).state_dict()
        elif options.save_stat == "pearson" and pearson_stat >= best_pearson:
            logger.log("Saving Model")
            best_pearson = pearson_stat
            best_model_weights = copy.deepcopy(model).state_dict()

        train_losses.append(avg_train_rmse)
        val_losses.append(avg_val_rmse)
        val_pearsons.append(pearson_stat)

        def rs(v: float) -> str:
            r = round(v, 4)
            return f"{r:.4f}"

        out_str = f"Epoch {epoch + 1}/{options.epochs}: Train RMSE: {rs(avg_train_rmse)} | Val RMSE: {rs(avg_val_rmse)} | Val Pearson: {pearson_stat}"
        out_str += f" | Best Diff: {rs(best_diff_e)} | Worst Diff: {rs(worst_diff_e)}"
        logger.log(out_str)

    model.load_state_dict(best_model_weights)
    model.eval()

    return train_losses, val_losses, val_pearsons


def train(configs: GenotypeToPhenotypeConfigs, options: TrainingOptions):
    # Initialize Logger
    exp_info = create_exp_info_text(
        species=options.species,
        wing=options.wing,
        color=options.color,
        chromosome=options.chromosome,
    )
    if options.top_k_chromosome_training:  # TODO handler top_k_chromosome training
        exp_info = f"{options.species}_{options.wing}_{options.color}_top_k_snps"

    training_output_dir = get_results_training_output_directory(configs.io)
    logger = ExperimentLogger(
        training_output_dir,
        exp_name=Path(options.exp_name, exp_info),
        log_fname="training",
        verbose=options.verbose,
    )
    print(f"Logging at: {logger.get_log_location()}")

    done_log_location = logger.get_log_location(log_name="DONE")
    if done_log_location.exists() and not options.force_retrain:
        print("Already trained this model")
        return

    # Load Training Data
    train_data, val_data, test_data = load_training_data(configs, options)

    num_vcfs = train_data[0].shape[1]
    logger.log(f"Input size: {num_vcfs}")
    logger.log(f"Number of out dimensions used: {options.out_dims}")

    train_dataset = GTP_Dataset(*train_data)
    val_dataset = GTP_Dataset(*val_data)
    test_dataset = GTP_Dataset(*test_data)

    train_dataloader = DataLoader(
        train_dataset, batch_size=options.batch_size, num_workers=8, shuffle=True
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=options.batch_size, num_workers=8, shuffle=False
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=options.batch_size, num_workers=8, shuffle=False
    )

    start_t = time.perf_counter()
    match options.model:
        case "soybean":
            model = SoyBeanNet(
                window_size=num_vcfs,
                num_out_dims=options.out_dims,
                insize=options.insize,
                hidden_dim=options.hidden_dim,
                drop_out_prob=options.drop_out_prob,
            ).cuda()
        case "deepnet":
            model = DeepNet(
                window_size=num_vcfs,
                num_out_dims=options.out_dims,
                insize=options.insize,
                hidden_dim=options.hidden_dim,
                drop_out_prob=options.drop_out_prob,
            ).cuda()

        case _:
            raise NotImplementedError(
                f"{options.model} is not a valid model. Please implement it or give a different option."
            )

    # Train the model
    train_losses, val_losses, val_pearsons = train_model(
        options, train_dataloader, val_dataloader, model=model, logger=logger
    )

    torch.save(model.state_dict(), Path(logger.outdir, "model.pt"))
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    logger.log(f"Total training time: {total_duration:.2f}s")

    logger.log("Testing")
    rmses = test(
        train_dataloader,
        val_dataloader,
        test_dataloader,
        model,
        options.out_dims,
        options.out_dims_start_idx,
    )
    logger.log(f"Train RMSE: {rmses[0]} | Val RMSE: {rmses[1]} | Test RMSE: {rmses[2]}")

    plot_loss_curves(train_losses, val_losses, logger.outdir)

    # TODO: significantly speed this up or just remove it from the training script
    # start_t = time.perf_counter()
    # logger.log("Beginning attribution")
    # for att_method in ["cam", "lrp"]:
    #    tr_pts, val_pts, test_pts = plot_attribution_graph(
    #        model,
    #        train_dataloader,
    #        val_dataloader,
    #        test_dataloader,
    #        logger.outdir,
    #        ignore_train=True,
    #        mode=att_method,
    #        ignore_plot=False,
    #        use_new=True,
    #    )
    #
    #    k = 2000
    #    if options.top_k_chromosome_training:
    #        k = num_vcfs
    #    plot_pvalue_filtering(test_pts, test_dataset, logger, prefix=att_method, k=k)
    #
    # end_t = time.perf_counter()
    # total_duration = end_t - start_t
    # logger.log(f"Total attribution time: {total_duration:.2f}s")

    logger.log("Completed!", log_name="DONE")


@click.command()
@click.option(
    "--configs",
    default=None,
    help="Path to YAML config file to be used in preprocessing.",
)
@TrainingOptions.click_options()
def main(configs, **kwargs):
    cfgs: GenotypeToPhenotypeConfigs = load_configs(configs)
    opts: TrainingOptions = TrainingOptions(**kwargs)

    train(cfgs, opts)


if __name__ == "__main__":
    main()
