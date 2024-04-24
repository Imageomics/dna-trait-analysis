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
from torch.utils.data import Dataset, DataLoader

import numpy as np

import matplotlib.pyplot as plt

from captum.attr import IntegratedGradients, NoiseTunnel, DeepLift, GradientShap, FeatureAblation, GuidedGradCam, Saliency, Occlusion

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

def save_loss_curve(train_losses, val_losses):
    rows = np.arange(len(train_losses))
    fig, ax = plt.subplots()
    ax.plot(rows, train_losses, label="training loss")
    ax.plot(rows, val_losses, label="validation loss")

    ax.set(xlabel='Epoch', ylabel='loss',
        title='Training curves')
    ax.grid()

    fig.savefig("loss_curves.png")
    plt.close()

#def get_top_k_pos(dataloader, model, K=5, target_pos=0):
#    target_layers = [model.last_block[-1]]
#    cam = GradCAM(model=model, target_layers=target_layers, use_cuda=True)
#    targets = [ClassifierOutputTarget(target_pos)]
#    scores = None
#    for name, data, pca in tqdm(test_dataloader, desc="Testing"):
#        for d in data:
#            sv = cam(input_tensor=d.unsqueeze(0).cuda(), targets=targets)
#            sv = sv.max(-1)[0]
#            if scores is None:
#                scores = sv[np.newaxis, :]
#            else:
#                scores = np.concatenate((scores, sv[np.newaxis, :]), axis=0)
#
#    median_scores = np.median(scores, axis=0)
#    top_k_pos = np.argsort(median_scores)[-K:]
#    return top_k_pos, median_scores[top_k_pos]

def forward_step(models, batch, optimizers, windows, loss_fn, args, is_train=True):
    name, data, pca = batch
    data = data.cuda()
    pca = pca[:, :args.out_dims].cuda()
    #pca = pca[:, 10:11].cuda()
    losses = []
    with torch.set_grad_enabled(is_train):
        for model, optimizer, window in zip(models, optimizers, windows):
            if is_train:
                model.train()
            else:
                model.eval()
            out = model(data[:, :, window-args.window_size:window, :])
            loss = loss_fn(out, pca)

            if is_train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            losses.append(loss.item())
        
        return losses



def train(args, dataloaders, models, windows):
    train_dataloader, val_dataloader, test_dataloader = dataloaders
    optimizers = [torch.optim.SGD(model.parameters(), lr=args.lr) for model in models]
    loss_fn = nn.MSELoss()
    #loss_fn = nn.L1Loss()

    best_models = []
    best_losses = []
    for epoch in tqdm(range(args.epochs), desc="Training Batch of models", colour="green"):
        
        # Training
        train_loss_totals = []
        for batch in train_dataloader:
            losses = forward_step(models, batch, optimizers, windows, loss_fn, args, is_train=True)

            if not train_loss_totals:
                train_loss_totals = [loss for loss in losses]
            else:
                train_loss_totals = [loss + prev for loss, prev in zip(losses, train_loss_totals)]
        
        train_loss_totals = [loss / len(train_dataloader) for loss in train_loss_totals]
        
        # Validation
        val_loss_totals = []
        for batch in val_dataloader:
            losses = forward_step(models, batch, optimizers, windows, loss_fn, args, is_train=False)
            if not val_loss_totals:
                val_loss_totals = [loss for loss in losses]
            else:
                val_loss_totals = [loss + prev for loss, prev in zip(losses, val_loss_totals)]

        val_loss_totals = [loss / len(val_dataloader) for loss in val_loss_totals]
        
        if not best_losses:
            best_losses = [loss for loss in val_loss_totals]
            best_models = [m.state_dict() for m in models]
        else:
            for i, (best_loss, val_loss) in enumerate(zip(best_losses, val_loss_totals)):
                if val_loss <= best_loss:
                    best_losses[i] = val_loss
                    best_models[i] = models[i].state_dict()

    
    for i, best_weights in enumerate(best_models):
        models[i].load_state_dict(best_weights)

    test_loss_totals = []
    for batch in test_dataloader:
        losses = forward_step(models, batch, optimizers, windows, loss_fn, args, is_train=False)
        if not test_loss_totals:
            test_loss_totals = [loss for loss in losses]
        else:
            test_loss_totals = [loss + prev for loss, prev in zip(losses, test_loss_totals)]

    test_loss_totals = [loss / len(test_dataloader) for loss in test_loss_totals]

    models[0].eval()
    models[0].zero_grad()
    ig = Occlusion(models[0])
    #ig = NoiseTunnel(ig)    
    with torch.no_grad():
        ig_attr_total = None
        for i, batch in enumerate(test_dataloader):
            name, data, pca = batch
            ig_attr = ig.attribute(data.cuda(), target=0, sliding_window_shapes=(1, 200, 3), strides=100, show_progress=True)
            ig_attr, _ = ig_attr.max(-1) # Max across 1-hot representation of input
            ig_attr = ig_attr[:, 0] # Only has 1 channel, just extract it
            ig_attr = ig_attr.sum(0) # Sum across batch
            if ig_attr_total is None:
                ig_attr_total = ig_attr.detach().cpu().numpy()
            else:
                ig_attr_total += ig_attr.detach().cpu().numpy()
    
    ig_attr_total = ig_attr_total / np.linalg.norm(ig_attr_total, ord=1) # Normalize
    
    plt.figure(figsize=(20, 10))

    ax = plt.subplot()
    ax.set_title('Comparing input feature importances across multiple algorithms and learned weights')
    ax.set_ylabel('Attributions')

    FONT_SIZE = 16
    plt.rc('font', size=FONT_SIZE)            # fontsize of the text sizes
    plt.rc('axes', titlesize=FONT_SIZE)       # fontsize of the axes title
    plt.rc('axes', labelsize=FONT_SIZE)       # fontsize of the x and y labels
    plt.rc('legend', fontsize=FONT_SIZE - 4)  # fontsize of the legend

    ax.bar(np.arange(len(ig_attr_total)), ig_attr_total, align='center', alpha=0.8, color='#eb5e7c')
    ax.autoscale_view()
    plt.tight_layout()

    plt.savefig("attribution.png")

    
    return train_loss_totals, best_losses, test_loss_totals

def get_norm_vals(data):
    out_data = np.array([d[2] for d in data])

    mu = out_data.mean(axis=0)
    std = out_data.std(axis=0)

    return mu, std


def get_args():
    parser = ArgumentParser()
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--out_dims", type=int, default=10)
    parser.add_argument("--insize", type=int, default=3)
    parser.add_argument("--window_size", type=str, default="200")
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--species", type=str, default="erato")
    parser.add_argument("--input_data", type=str, default="/local/scratch/carlyn.1/dna/vcfs/erato_dna_matrix.npz")
    parser.add_argument("--input_metadata", type=str, default="/local/scratch/carlyn.1/dna/vcfs/erato_names.json")
    parser.add_argument("--pca_loadings", type=str, default="/local/scratch/carlyn.1/dna/colors/erato_red_loadings.csv")
    parser.add_argument("--exp_name", type=str, default="test")
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--num_models", type=int, default=8)

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    input_data = np.load(args.input_data)['arr_0']
    num_vcfs = input_data.shape[1]

    if args.window_size == "full":
        args.window_size = num_vcfs
    else:
        args.window_size = int(args.window_size)

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

    windows = list(range(args.window_size, num_vcfs, args.window_size)) + [num_vcfs]
    start_t = time.perf_counter()

    cur_idx = 0
    while cur_idx < len(windows):
        start_t_b = time.perf_counter()
        add_amt = args.num_models
        if (cur_idx + add_amt) >= len(windows):
            add_amt = len(windows) - cur_idx
        
        train_windows = windows[cur_idx:cur_idx+add_amt]
        models = []
        for _ in range(add_amt):
            #model = ConvNet(num_out_dims=out_dims).cuda()
            models.append(SoyBeanNet(window_size=args.window_size, num_out_dims=args.out_dims, insize=args.insize).cuda())
            #model = LargeNet(num_out_dims=out_dims).cuda()
        
        train_losses, val_losses, test_losses = train(args, [train_dataloader, val_dataloader, test_dataloader], models=models, windows=train_windows)
        results_file_path = os.path.join(args.output_dir, f"{args.exp_name}_sliding_window.txt")
        with open(results_file_path, 'a') as f:
            for i, (train_loss, val_loss, test_loss) in enumerate(zip(train_losses, val_losses, test_losses)):
                out_str = f"Window: [{train_windows[i]-args.window_size}, {train_windows[i]}] | Train Loss ({train_loss}) | Val Loss ({val_loss}) | Test Loss ({test_loss})"
                print(out_str)
                f.write(out_str + '\n')
        
        create_curve(results_file_path, os.path.join(args.output_dir, f"{args.exp_name}_curve.png"))

        cur_idx += add_amt

        end_t_b = time.perf_counter()
        total_duration = end_t_b - start_t_b
        print(f"Total batch training time: {total_duration:.2f}s")

    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"Total training time: {total_duration:.2f}s")



