import numpy as np
from torch.utils.data import Dataset, DataLoader

class CustomDataset(Dataset):
    def __init__(self, DE, CE, TCI, CKE, is_train = True, FakeRand = False, shuffle = True):
        self.DE = DE
        self.CE = CE
        self.TCI = TCI
        self.CKE = CKE
        self.is_train = is_train
        self.FakeRand = FakeRand
        self.shuffle = True

        self.ClassEmbedding_with1 = np.c_[self.CE, np.ones(len(self.CE))]
        self.TrainCandidateKinases_with1 = np.c_[self.CKE, np.ones(len(self.CKE))]


    def __len__(self):
        return len(self.DE)

    def __getitem__(self, idx):
        return self.DE[idx], self.CE[idx], self.TCI[idx]

    def _next_batch(self, start, epochs_completed, batch_size):

        # Shuffle for the first epoch
        if epochs_completed == 0 and start == 0 and self.shuffle:
            perm0 = np.arange(len(self.DE))
            if self.FakeRand:
                np.random.RandomState(42).shuffle(perm0)
            else:
                np.random.shuffle(perm0)
            self.DE_Shuffled = self.DE[perm0]
            self.CE_Shuffled = self.CE[perm0]
            self.TCI_Shuffled = self.TCI[perm0]

        if start + batch_size > len(self.DE):
          # Finished epoch
          epochs_completed += 1
          # Get the rest examples in this epoch
          rest_num_examples = len(self.DE) - start
          DE_rest_part = self.DE[start:len(self.DE)]
          CE_rest_part = self.CE[start:len(self.CE)]
          TCI_rest_part = self.TCI[start:len(self.TCI)]
          # Shuffle the data
          if self.shuffle:
            perm = np.arange(len(self.DE))
            if self.FakeRand:
                np.random.RandomState(17).shuffle(perm)
            else:
                np.random.shuffle(perm)
            self.DE_Shuffled = self.DE[perm]
            self.CE_Shuffled = self.CE[perm]
            self.TCI_Shuffled = self.TCI[perm]
          # Start next epoch
          start_new = 0
          new_start_index = batch_size - rest_num_examples
          end = new_start_index
          DE_new_part = self.DE_Shuffled[start_new:end]
          CE_new_part = self.CE_Shuffled[start_new:end]
          TCI_new_part = self.TCI_Shuffled[start_new:end]
          return np.concatenate((DE_rest_part, DE_new_part), axis=0), np.concatenate((CE_rest_part, CE_new_part), axis=0), np.concatenate((TCI_rest_part, TCI_new_part), axis=0),new_start_index,epochs_completed
        else:
          new_start_index = start + batch_size
          end = new_start_index
          return self.DE_Shuffled[start:end], self.CE_Shuffled[start:end], self.TCI_Shuffled[start:end],new_start_index,epochs_completed