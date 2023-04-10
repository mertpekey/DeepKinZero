import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

# X, ClassEmbedding, TrueClassIDX, TrainCandidateKinases

# Hepsini Torch.Tensor'e cevir
class DKZ_Dataset(Dataset):
    def __init__(self, DE, CE, TCI, CKE, is_train = True):

        self.DE = torch.from_numpy(DE).float() # Phosphosite (12901, 13, 100)
        if is_train:
            self.CE = torch.from_numpy(CE).float() # Kinase Embedding (12901, 727)
            self.TCI = torch.from_numpy(TCI) # Labels (12901)
            self.CKE = torch.from_numpy(CKE) # Unique Kinase Embedding (214,727)
        else:
            self.CE = CE
            self.TCI = TCI
            self.CKE = CKE
        self.is_train = is_train

        if is_train:
            self.ClassEmbedding_with1 = torch.from_numpy(np.c_[self.CE, np.ones(len(self.CE))]).float() # (12901, 728)
            self.TrainCandidateKinases_with1 = torch.from_numpy(np.c_[self.CKE, np.ones(len(self.CKE))]).float() # (214,728)
        else:
            self.ClassEmbedding_with1 = np.c_[self.CE, np.ones(len(self.CE))]
            self.TrainCandidateKinases_with1 = np.c_[self.CKE, np.ones(len(self.CKE))]

    def __len__(self):
        return len(self.DE)

    def __getitem__(self, idx):
        #return self.DE[idx], self.ClassEmbedding_with1[idx], self.TCI[idx]
        return self.DE[idx], self.ClassEmbedding_with1[idx], self.TCI[idx]
