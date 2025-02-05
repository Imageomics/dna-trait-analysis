import os
from multiprocessing import Pool

import matplotlib.pyplot as plt
import numpy as np
import torch
from captum.attr import LRP, GuidedGradCam, Occlusion, Saliency, ShapleyValueSampling
from scipy.stats import pearsonr
from tqdm import tqdm

from gtp.models.forward import forward_step


def do_knockout(model, dloader, target=0, out_dims=1):
    def knockout(batch, pos):
        name, data, pca = batch
        # knockout
        data[:, :, pos] = torch.zeros_like(data[:, :, pos]).to(data.device)
        return name, data, pca

    for batch in dloader:
        dim_size = batch[1].shape[2]
        break
    knockout_values = []
    print(f"DIM SIZE: {dim_size}")
    for pos in tqdm(range(dim_size), desc="Performing knockout"):
        total_rmse = 0
        for batch in dloader:
            batch = knockout(batch, pos)
            mse, rmse = forward_step(model, batch, None, out_dims, is_train=False)
            total_rmse += rmse

        avg_rmse = total_rmse / len(dloader)
        knockout_values.append(avg_rmse)

    return knockout_values


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


def calc_pearson_correlation(model, dloader, index=0):
    model.eval()
    actual = []
    predicted = []
    for i, batch in enumerate(dloader):
        model.zero_grad()
        name, data, pca = batch
        out = model(data.cuda())
        actual.extend(pca[:, index].detach().cpu().numpy().tolist())
        predicted.extend(out[:, index].detach().cpu().numpy().tolist())
    return pearsonr(predicted, actual)


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


def get_lrp_attr(m, dloader, target=0, verbose=False, num_processes=1):
    att_model = LRP(m)
    all_actuals = []
    all_relevance_scores = []
    for i, batch in tqdm(
        enumerate(dloader), desc="Calculating LRP Attributions", disable=not verbose
    ):
        m.zero_grad()
        data, pca = batch
        data.requires_grad = True
        attr = att_model.attribute(data.cuda(), target=target)
        # For LRP, this should be sum. This is because the attribution scores should all add up to be the find value in the prediction, so averaging could break that.
        attr = attr.sum(-1)
        attr = attr[:, 0]  # Only has 1 channel, just extract it
        attr = attr.detach().cpu().numpy()

        all_relevance_scores.append(attr)
        all_actuals.append(pca[:, target].detach().cpu().numpy())

    all_relevance_scores = np.concatenate(all_relevance_scores, axis=0)
    all_actuals = np.concatenate(all_actuals, axis=0)
    average_relevance_scores = np.abs(all_relevance_scores).mean(0)

    process_results = []
    N = all_relevance_scores.shape[1]
    with (
        Pool(processes=num_processes) as p,
        tqdm(
            total=N,
            desc="Processing Genotype data",
            colour="#87ceeb",  # Skyblue
            disable=not verbose,
        ) as pbar,
    ):
        process_data = [
            (
                i,
                np.concatenate(
                    (all_relevance_scores[:, i : i + 1], all_actuals[:, np.newaxis]),
                    axis=1,
                ),
            )
            for i in range(N)
        ]
        for result in p.imap_unordered(_lrp_pearson_stat_multi_fn, process_data):
            process_results.append(result)
            pbar.update()
            pbar.refresh()

    snp_pearson_stats = [x[1:] for x in sorted(process_results, key=lambda x: x[0])]

    return np.concatenate(
        (
            average_relevance_scores[:, np.newaxis],
            np.array(snp_pearson_stats).astype(np.float64),
        ),
        axis=1,
    )


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
