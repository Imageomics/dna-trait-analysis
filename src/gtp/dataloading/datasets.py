import torch
from torch.utils.data import Dataset

class GTP_Dataset(Dataset):
    def __init__(self, genotype_data, phenotype_data):
        super().__init__()
        self.genotype_data = genotype_data
        self.phenotype_data = phenotype_data

    def __getitem__(self, idx):
        x = self.genotype_data[idx]
        y = self.phenotype_data[idx]
        
        in_data = torch.tensor(x).unsqueeze(0).float()
        out_data = torch.tensor(y).float()

        return in_data, out_data

    def __len__(self):
        return len(self.genotype_data)