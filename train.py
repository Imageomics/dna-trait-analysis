import os
import json
import random

from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import numpy as np

from data_tools import parse_patternize_csv

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

class VCF_Dataset(Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, idx):
        name, in_data, out_data = self.data[idx]
        in_data = torch.tensor(in_data).unsqueeze(0).float()
        out_data = torch.tensor(out_data).float()

        return name, in_data, out_data

    def __len__(self):
        return len(self.data)

class ConvNet(nn.Module):
    def __init__(self, num_out_dims=10):
        super().__init__()

        self.block1 = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(16, 16, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0)
        )
        
        self.block2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(32, 32, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0)
        )

        self.block3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0)
        )
        
        self.block4 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0)
        )
        
        self.block5 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0),
            nn.Conv2d(64, 64, kernel_size=3, padding=1, stride=1),
            nn.LeakyReLU(0.2),
            nn.MaxPool2d((2, 1), stride=(2, 1), padding=0)
        )

        self.fc = nn.Linear(64*4*56, num_out_dims)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = x.view(len(x), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    # Load VCFS
    VCF_OUTPUT = "/local/scratch/carlyn.1/dna/vcfs/"
    input_data = np.load(os.path.join(VCF_OUTPUT, "erato_dna_matrix.npz"))['arr_0']
    metadata = load_json(os.path.join(VCF_OUTPUT, "erato_names.json"))

    # Load PCA values
    PCA_OUTPUT = "/local/scratch/carlyn.1/dna/colors/erato_red_loadings.csv"
    pca_data = parse_patternize_csv(PCA_OUTPUT)

    train_data = []

    for name, row in zip(metadata, input_data):
        if name+"_d" in pca_data:
            train_data.append([name, row, pca_data[name+"_d"]])


    random.seed(2)
    random.shuffle(train_data)
    train_idx = int(len(train_data) * 0.8)
    val_idx = int(len(train_data) * 0.1)

    train_split = train_data[:train_idx]
    val_split = train_data[train_idx:train_idx+val_idx]
    test_split = train_data[train_idx+val_idx:]

    train_dataset = VCF_Dataset(train_split)
    val_dataset = VCF_Dataset(val_split)
    test_dataset = VCF_Dataset(test_split)
    
    out_dims = 20
    model = ConvNet(num_out_dims=out_dims).cuda()

    #model_path = "model.pt"
    #if os.path.isfile(model_path):
    #    model.load_state_dict(torch.load(model_path))

    train_dataloader = DataLoader(train_dataset, batch_size=64, num_workers=4, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, num_workers=4, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=64, num_workers=4, shuffle=False)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    loss_fn = nn.MSELoss()

    best_val_loss = 9999999
    for epoch in range(100):
        loss_total = 0
        model.train()
        for name, data, pca in tqdm(train_dataloader, desc="Training"):
            out = model(data.cuda())
            loss = loss_fn(out, pca[:, :out_dims].cuda())
            loss_total += loss.item()

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        val_loss = 0
        model.eval()
        with torch.no_grad():
            for name, data, pca in tqdm(val_dataloader, desc="Validation"):
                out = model(data.cuda())
                loss = loss_fn(out, pca[:, :out_dims].cuda())
                val_loss += loss.item()

        if val_loss <= best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "model.pt")

        print(f"Epoch {epoch+1}: Train Loss => {loss_total/len(train_dataloader)} | Val Loss => {val_loss/len(val_dataloader)}")

    model.load_state_dict(torch.load("model.pt"))
    val_loss = 0
    model.eval()
    with torch.no_grad():
        for name, data, pca in tqdm(test_dataloader, desc="Testing"):
            out = model(data.cuda())
            loss = loss_fn(out, pca[:, :out_dims].cuda())
            val_loss += loss.item()
    print(f"Epoch {epoch+1}: Test Loss => {val_loss/len(test_dataloader)}")


