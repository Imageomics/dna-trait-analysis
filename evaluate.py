import time
import os
from argparse import ArgumentParser

import numpy as np

import torch
from torch.utils.data import DataLoader

from data_tools import load_json, parse_patternize_csv, VCF_Dataset
from models.net import SoyBeanNet
from evaluation import test, get_attribution_points

def load_data(args):
    input_data = np.load(args.input_data)['arr_0']

    metadata = load_json(args.input_metadata)
    pca_data = parse_patternize_csv(args.pca_loadings)

    # Split data originally
    split_lists = {}
    for split_type in ["train", "val", "test"]:
        split_lists[split_type] = []
        fpath = os.path.join(args.results_dir, args.exp_name, f"{split_type}_split.txt")
        with open(fpath, 'r') as f:
            for line in f.readlines():
                line = line.strip()
                data_ids = line.split(",")
                split_lists[split_type].extend(data_ids)
    
        #print(split_lists[split_type])

    # Assert that the intersection of all the splits is empty
    #print(set(split_lists["train"]) & set(split_lists["val"]))
    assert not set(split_lists["train"]) & set(split_lists["val"]), "Common element in train and val sets"
    assert not set(split_lists["val"]) & set(split_lists["test"]), "Common element in val and test sets"
    assert not set(split_lists["train"]) & set(split_lists["test"]), "Common element in train and test sets"

    data_splits = {"train": [], "val": [], "test": []}
    for name, row in zip(metadata, input_data):
        if name+"_d" in pca_data:
            if name in split_lists["train"]:
                data_splits["train"].append([name, row, pca_data[name+"_d"]])
            elif name in split_lists["val"]:
                data_splits["val"].append([name, row, pca_data[name+"_d"]])
            elif name in split_lists["test"]:
                data_splits["test"].append([name, row, pca_data[name+"_d"]])

    num_vcfs = data_splits["train"][0][1].shape[0]

    train_dataset = VCF_Dataset(data_splits["train"])
    val_dataset = VCF_Dataset(data_splits["val"])
    test_dataset = VCF_Dataset(data_splits["test"])

    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=8, shuffle=False)

    return train_dataloader, val_dataloader, test_dataloader, num_vcfs

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=512)
    parser.add_argument("--out_dims", type=int, default=50)
    parser.add_argument("--insize", type=int, default=3)
    parser.add_argument("--seed", type=int, default=2)
    parser.add_argument("--input_data", type=str, default="/local/scratch/carlyn.1/dna/vcfs/erato_dna_matrix.npz")
    parser.add_argument("--input_metadata", type=str, default="/local/scratch/carlyn.1/dna/vcfs/erato_names.json")
    parser.add_argument("--pca_loadings", type=str, default="/local/scratch/carlyn.1/dna/colors/erato_red_loadings.csv")
    parser.add_argument("--results_dir", type=str, default="results")
    parser.add_argument("--exp_name", type=str, default="test")

    return parser.parse_args()

def save_points(train_pts, val_pts, test_pts, args):
    np.savez(os.path.join(args.results_dir, args.exp_name, "att_points.npz"), train=train_pts, val=val_pts, test=test_pts)

if __name__ == "__main__":
    args = get_args()

    # Get data
    train_dataloader, val_dataloader, test_dataloader, num_vcfs = load_data(args)
    
    # Get model
    model = SoyBeanNet(window_size=num_vcfs, num_out_dims=args.out_dims, insize=args.insize).cuda()
    model.load_state_dict(torch.load(os.path.join(os.path.join(args.results_dir, args.exp_name, "model.pt"))))

    # Get RMSE
    rmses = test(train_dataloader, val_dataloader, test_dataloader, model, args.out_dims)
    print(f"Train RMSE: {rmses[0]} | Val RMSE: {rmses[1]} | Test RMSE: {rmses[2]}")

    # Get Plot points for each loader
    start_t = time.perf_counter()
    print("Beginning attribution")
    model.eval()
    train_att_points = get_attribution_points(model, train_dataloader)
    val_att_points = get_attribution_points(model, val_dataloader)
    test_att_points = get_attribution_points(model, test_dataloader)
    save_points(train_att_points, val_att_points, test_att_points, args)
    end_t = time.perf_counter()
    total_duration = end_t - start_t
    print(f"Total attribution time: {total_duration:.2f}s")