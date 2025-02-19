import os
from multiprocessing import Pool
from enum import Enum

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.nn.functional import l1_loss, mse_loss
from captum.attr import LRP, GuidedGradCam, Occlusion, Saliency, ShapleyValueSampling
from scipy.stats import pearsonr
from tqdm import tqdm

from gtp.models.forward import forward_step


class AttributionMethod(Enum):
    LRP = "lrp"  # Layerwise Relevance Propagation
    PERTURB = "perturb"  # Perturbation on the DNA input states


def test(tr_dloader, val_dloader, test_dloader, model, out_dims, out_dims_start_idx=0):
    rmses = []
    for dl in [tr_dloader, val_dloader, test_dloader]:
        total_rmse = 0
        for batch in dl:
            mse, rmse = forward_step(
                model,
                batch,
                None,
                out_dims,
                out_start_idx=out_dims_start_idx,
                is_train=False,
            )
            total_rmse += rmse

        avg_rmse = total_rmse / len(dl)
        rmses.append(avg_rmse)

    return rmses


def compile_attribution(attr):
    attr, _ = torch.abs(attr).max(-1)
    attr = attr[:, 0]  # Only has 1 channel, just extract it
    attr = attr.detach().cpu().numpy()
    attr = np.abs(attr).sum(0)  # Sum across batch...Should we be summing here???
    # attr = attr.sum(0) # Sum across batch...Should we be summing here???
    return attr


def compile_attribution_test(attr):
    # attr, _ = torch.abs(attr).max(-1)
    attr, _ = torch.abs(attr).max(-1)
    attr = attr[:, 0]  # Only has 1 channel, just extract it
    attr = attr.detach().cpu().numpy()
    attr = np.median(np.abs(attr), 0)  # Sum across batch...Should we be summing here???
    # attr = attr.sum(0) # Sum across batch...Should we be summing here???
    return attr


def get_attribution_points(model, dloader, target=0):
    att_model = Occlusion(model)
    attr_total = None
    for i, batch in enumerate(dloader):
        model.zero_grad()
        data, pca = batch
        attr = att_model.attribute(
            data.cuda(),
            target=target,
            sliding_window_shapes=(1, 200, 3),
            strides=20,
            show_progress=True,
        )
        attr = compile_attribution(attr)
        if attr_total is None:
            attr_total = attr
        else:
            attr_total += attr

    # attr_total = attr_total / np.linalg.norm(attr_total, ord=1) # Normalize
    return attr_total


def get_shapley_sampling_attr(m, dloader, target=0, n_samples=200, n_windows=100):
    svs = ShapleyValueSampling(m)
    WINDOWS = n_windows
    feature_mask = None
    attr_total = None
    for i, batch in enumerate(dloader):
        m.zero_grad()
        name, data, pca = batch

        if feature_mask is None:
            feature_mask = torch.zeros_like(data[0]).unsqueeze(0)
            ws = data.shape[2] // WINDOWS
            for j in range(0, WINDOWS, ws):
                feature_mask[:, :, j * ws :, :] = j
            feature_mask = feature_mask.cuda()

        attr = svs.attribute(
            data.cuda(),
            feature_mask=feature_mask,
            target=target,
            n_samples=n_samples,
            show_progress=True,
        )
        attr = compile_attribution(attr)
        if attr_total is None:
            attr_total = attr
        else:
            attr_total += attr

    return attr_total


def _lrp_pearson_stat_multi_fn(process_item):
    snp_pos, data = process_item
    pearson_results = pearsonr(data[:, 0], data[:, 1])
    pcc = pearson_results.statistic
    pvalue = pearson_results.pvalue
    return snp_pos, pcc, pvalue


def get_lrp_attr(m, dloader, targets=0, verbose=False, num_processes=1):
    if isinstance(targets, int):
        targets = [targets]

    att_model = LRP(m)
    all_actuals = []
    all_relevance_scores = []
    for i, batch in tqdm(
        enumerate(dloader), desc="Calculating LRP Attributions", disable=not verbose
    ):
        m.zero_grad()
        data, pca = batch
        data.requires_grad = True
        all_attrs = []
        for tgt in targets:
            attr = att_model.attribute(data.cuda(), target=tgt)
            all_attrs.append(attr.detach().cpu())
        all_attrs = torch.stack(all_attrs, dim=0)
        # For LRP, this (ONE-HOT state ex. [0,0,1] attributions) should be sum.
        # This is because the attribution scores should all add up to be the find value in the prediction, so averaging could break that.
        all_attrs = all_attrs.sum(-1)
        all_attrs = all_attrs[:, :, 0]  # Only has 1 channel, just extract it
        all_attrs = (
            all_attrs.detach().cpu().numpy()
        )  # Num-Targets x Batch-Size x Input-Dimension

        all_relevance_scores.append(all_attrs)

    all_relevance_scores = np.concatenate(all_relevance_scores, axis=1)
    average_relevance_scores = np.abs(all_relevance_scores).mean(1)

    return average_relevance_scores


def get_perturb_attr(m, dloader, targets=0, distance_fn="l1", verbose=False):
    """This attribution method will record the change in the output (y) when changing the state of the input at each
    feature location.

    VERY SLOW at the moment. TODO: speed this up

    IMPORTANT: ensure the dataloader passed in has shuffle turned off for proper alignment!

    Algorithm:
    1. Each datapoint will be passed through the model to obtain baseline outputs (base_y)
    2. For each feature dimension, for all inputs:
        2a. The state will be shifted (AA -> Aa/aA, Aa/aA -> aa, aa -> AA)
        2b. These shifted states will be passed through the model to obtain a perturbation affected output (perturb_y)
        2c. We record the largest variance of the change in the output (perturb_y - base_y)
    3. Average the magnitude of the variance change across samples (N)
    4. Return average magnitude of each dimension (D)

    Args:
        m (_type_): The predictive model
        dloader (_type_): The dataloader that returns the intput X and output y
        targets (List[int], optional): The indexs of the output for the desired attribution. Defaults to 0.
        distance_fn (str, optional): The distance function to calculate the perturbed effect. Defaults to l1.
        verbose (bool, optional): Set True if everything is to be printed. Defaults to False.

    Returns:
        _type_: The resulting attributions per input dimension D
    """
    if isinstance(targets, int):
        targets = [targets]
    m.eval()

    with torch.no_grad():
        all_max_changes = None  # Should be N x D
        for i, batch in tqdm(
            enumerate(dloader),
            desc="Calculating perturbation attributions",
            colour="#87ceeb",
            disable=not verbose,
            leave=True,
            position=1,
        ):
            data, pca = batch  # data => B x D x 3
            all_attrs = []
            all_attrs = torch.stack(all_attrs, dim=0)
            output = m(data.cuda())[:, targets]
            base_y = output.detach().cpu()
            B, C, D, _ = data.shape  # batch x 1 x dimension x 3
            batch_max_magnitudes = np.zeros((B, D))
            for d in tqdm(
                range(D), desc="Perturbing batch", colour="red", position=0, leave=False
            ):
                data_perturbed = data.detach().clone()
                outputs_from_shifts = None
                for shift in range(2):
                    data_perturbed[:, :, d, :] = data_perturbed[:, :, d, :][
                        :, :, torch.tensor([2, 0, 1])
                    ]  # Shift columns to the right. this simulates a state change if the input is 1-hot
                    perturbed_output = (
                        m(data_perturbed.cuda())[:, targets].detach().cpu()
                    )
                    if distance_fn == "l1":
                        diff = l1_loss(perturbed_output, base_y)
                    elif distance_fn in ["l2", "mse"]:
                        diff = mse_loss(perturbed_output, base_y)
                    variance_magnitude = diff.unsqueeze(-1)
                    if outputs_from_shifts is None:
                        outputs_from_shifts = variance_magnitude
                    else:
                        outputs_from_shifts = torch.cat(
                            (outputs_from_shifts, variance_magnitude), dim=1
                        )
                # Record max of the two state changes
                batch_max_magnitudes[:, d] = torch.max(outputs_from_shifts, dim=1)[0]

            if all_max_changes is None:
                all_max_changes = batch_max_magnitudes
            else:
                all_max_changes = torch.cat(
                    (all_max_changes, batch_max_magnitudes), dim=0
                )

    averaged_max_changes = all_max_changes.mean(0)  # D
    return averaged_max_changes


# TODO: need to refactor, don't need new_new
def get_guided_gradcam_attr(m, dloader, target=0, use_new=False):
    att_model = GuidedGradCam(m, m.last_block)
    attr_total = None
    for i, batch in enumerate(dloader):
        m.zero_grad()
        if use_new:
            data, pca = batch
        else:
            name, data, pca = batch
        attr = att_model.attribute(data.cuda(), target=target)
        attr = compile_attribution(attr)
        if attr_total is None:
            attr_total = attr
        else:
            attr_total += attr

    # attr_total = attr_total / np.linalg.norm(attr_total, ord=1) # Normalize
    return attr_total


def get_guided_gradcam_attr_test(m, dloader, target=0):
    att_model = GuidedGradCam(m, m.last_block)
    attr_total = None
    for i, batch in enumerate(dloader):
        m.zero_grad()
        name, data, pca = batch
        attr = att_model.attribute(data.cuda(), target=target)
        attr = compile_attribution_test(attr)
        if attr_total is None:
            attr_total = attr
        else:
            attr_total += attr

    # attr_total = attr_total / np.linalg.norm(attr_total, ord=1) # Normalize
    return attr_total


def get_saliency_attr(m, dloader, target=0):
    att_model = Saliency(m)
    attr_total = None
    for i, batch in enumerate(dloader):
        m.zero_grad()
        name, data, pca = batch
        attr = att_model.attribute(data.cuda(), target=target)
        attr = compile_attribution(attr)
        if attr_total is None:
            attr_total = attr
        else:
            attr_total += attr

    return attr_total


def plot_attribution_graph(
    model,
    train_dataloader,
    val_dataloader,
    test_dataloader,
    outdir,
    ignore_train=False,
    mode="cam",
    ignore_plot=False,
    use_new=False,
):
    model.eval()
    # att_m = ShapleyValueSampling(model)

    all_att_pts = []
    for dloader, run_type in zip(
        [train_dataloader, val_dataloader, test_dataloader], ["train", "val", "test"]
    ):
        if ignore_train and run_type == "train":
            all_att_pts.append([])
            continue

        if mode == "cam":
            attr_total = get_guided_gradcam_attr(model, dloader, use_new=use_new)
        elif mode == "occlusion":
            with torch.no_grad():
                attr_total = get_attribution_points(model, dloader)
        elif mode == "lrp":
            attr_total = get_lrp_attr(model, dloader)
        else:
            raise NotImplementedError(
                f"Attribution mode ({mode}) has not been implemented"
            )

        # TODO: need to implement the moving window as in the deepcombi paper

        all_att_pts.append(attr_total)
        if not ignore_plot:
            for i in range(2):
                plt.figure(figsize=(20, 10))
                ax = plt.subplot()
                ax.set_title("Occlusion attribution of each SNP")
                ax.set_ylabel("Attributions")

                FONT_SIZE = 16
                plt.rc("font", size=FONT_SIZE)  # fontsize of the text sizes
                plt.rc("axes", titlesize=FONT_SIZE)  # fontsize of the axes title
                plt.rc("axes", labelsize=FONT_SIZE)  # fontsize of the x and y labels
                plt.rc("legend", fontsize=FONT_SIZE - 4)  # fontsize of the legend

                print("Plotting")
                y = attr_total
                if i == 1:
                    y = np.abs(attr_total)
                ax.bar(
                    np.arange(len(attr_total)),
                    y,
                    align="center",
                    alpha=0.8,
                    color="#eb5e7c",
                )
                print("End Plotting")
                # ax.autoscale_view()
                plt.tight_layout()
                fname = f"{run_type}_attribution"
                if i == 1:
                    fname += "_abs"
                plt.savefig(os.path.join(outdir, f"{fname}.png"))
                plt.close()

    np.savez(
        os.path.join(outdir, "att_points.npz"),
        train=all_att_pts[0],
        val=all_att_pts[1],
        test=all_att_pts[2],
    )

    return all_att_pts
