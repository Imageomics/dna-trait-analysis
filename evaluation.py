import os

import numpy as np

import torch

import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients, NoiseTunnel, DeepLift, GradientShap, FeatureAblation, GuidedGradCam, Saliency, Occlusion, ShapleyValueSampling
from models.forward import forward_step


def test(tr_dloader, val_dloader, test_dloader, model, out_dims):
    rmses = []
    for dl in [tr_dloader, val_dloader, test_dloader]:
        total_rmse = 0
        for batch in dl:
            mse, rmse = forward_step(model, batch, None, out_dims, is_train=False)
            total_rmse += rmse

        avg_rmse = total_rmse / len(tr_dloader)
        rmses.append(avg_rmse)

    return rmses

def get_attribution_points(model, dloader):
    att_model = Occlusion(model)
    attr_total = None
    for i, batch in enumerate(dloader):
        model.zero_grad()
        name, data, pca = batch
        attr = att_model.attribute(data.cuda(), target=0, sliding_window_shapes=(1, 200, 3), strides=20, show_progress=True)
        #attr = att_m.attribute(data.cuda(), target=0, show_progress=True)
        #attr = attr.abs() # Just take the abs value
        #attr, _ = attr.max(-1) # Max across 1-hot representation of input
        attr = attr.sum(-1)
        attr = attr[:, 0] # Only has 1 channel, just extract it
        attr = attr.sum(0) # Sum across batch
        if attr_total is None:
            attr_total = attr.detach().cpu().numpy()
        else:
            attr_total += attr.detach().cpu().numpy()

    attr_total = attr_total / np.linalg.norm(attr_total, ord=1) # Normalize
    return attr_total


def plot_attribution_graph(model, train_dataloader, val_dataloader, test_dataloader, outdir):
    model.eval() 
    #att_m = ShapleyValueSampling(model)  
    with torch.no_grad():
        all_att_pts = []
        for dloader, run_type in zip([train_dataloader, val_dataloader, test_dataloader], ["train", "val", "test"]):
            attr_total = get_attribution_points(model, dloader)
            all_att_pts.append(attr_total)
            for i in range(2):
                plt.figure(figsize=(20, 10))
                ax = plt.subplot()
                ax.set_title('Occlusion attribution of each SNP')
                ax.set_ylabel('Attributions')

                FONT_SIZE = 16
                plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
                plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
                plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
                plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

                print("Plotting")
                y = attr_total
                if i == 1:
                    y = np.abs(attr_total)
                ax.bar(np.arange(len(attr_total)), y, align='center', alpha=0.8, color='#eb5e7c')
                print("End Plotting")
                #ax.autoscale_view()
                plt.tight_layout()
                fname = f"{run_type}_attribution"
                if i == 1:
                    fname += "_abs"
                plt.savefig(os.path.join(outdir, f"{fname}.png"))
                plt.close()

    np.savez(os.path.join(outdir, "att_points.npz"), train=all_att_pts[0], val=all_att_pts[1], test=all_att_pts[2])