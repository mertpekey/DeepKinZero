import numpy as np
import torch
from torch.utils.data import Dataset

# X, ClassEmbedding, TrueClassIDX, TrainCandidateKinases

# Hepsini Torch.Tensor'e cevir
class DKZ_Dataset(Dataset):
    def __init__(self, DE, CE, TCI, CKE, args, is_train = True, is_input_tensor=False):

        self.args = args

        if is_input_tensor:
            self.phosphosite = DE
        else:
            self.phosphosite = torch.from_numpy(DE).float() # Phosphosite Protvec: (12901, 13, 100), Huggingface:
            
        if is_train:
            self.kinase = torch.from_numpy(CE).float() # Kinase Embedding (12901, 727)
            self.labels = torch.from_numpy(TCI) # Labels (12901)
            self.kinase_set = torch.from_numpy(CKE) # Unique Kinase Embedding (214,727)
        else:
            self.kinase = CE
            self.labels = TCI
            self.kinase_set = CKE
        self.is_train = is_train
        
        if is_train:
            self.kinase_with_1 = torch.from_numpy(np.c_[self.kinase, np.ones(len(self.kinase))]).float() # (12901, 728)
            self.kinase_set_with_1 = torch.from_numpy(np.c_[self.kinase_set, np.ones(len(self.kinase_set))]).float() # (214,728)
        else:
            self.kinase_with_1 = np.c_[self.kinase, np.ones(len(self.kinase))]
            self.kinase_set_with_1 = np.c_[self.kinase_set, np.ones(len(self.kinase_set))]

    def __len__(self):
        return len(self.phosphosite)

    def __getitem__(self, idx):
        #return self.DE[idx], self.ClassEmbedding_with1[idx], self.TCI[idx]
        if self.args.HF_ONLY_ID:
            return self.phosphosite[idx], self.kinase_with_1[idx], self.labels[idx]
        else:
            return {'input_ids': self.phosphosite['input_ids'][idx],
                    'attention_mask': self.phosphosite['attention_mask'][idx]}, self.kinase_with_1[idx], self.labels[idx]
