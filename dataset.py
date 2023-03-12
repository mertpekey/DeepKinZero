import torch
from torch.utils.data import Dataset, DataLoader

class KinaseDataset(Dataset):
    
    def __init__(self, data, transform=None, is_family=False):
        
        self.transform = transform # Scaler
        self.data = data # Kinase

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        pass