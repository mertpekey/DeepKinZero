import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

from model import softmax
from utils import GetAccuracyMultiLabel
import config as config


class Trainer:
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99954, last_epoch=-1)
        self._start = 0
        self._epochs_completed = 0

        # Set model device
        self.model.to(self.device)


    def train_step(self, train_dataset):
        self.model.train()

        # Get Batch (Change this with dataloader)
        batch_Xs, batch_CEs, batch_TCIs, self._start, self._epochs_completed = train_dataset._next_batch(self._start, self._epochs_completed, config.BATCH_SIZE)

        # Set device
        batch_Xs.to(self.device)
        batch_CEs.to(self.device)
        batch_TCIs.to(self.device)
        train_dataset.TrainCandidateKinases_with1.to(self.device)
        # Zero the gradients
        self.optimizer.zero_grad()

        # Get model prediction
        pred = self.model(batch_Xs.float())
        
        # Calculate F = DE * W * CE for all the CEs in unique class embeddings (all the kinases)
        logits = torch.matmul(pred, train_dataset.TrainCandidateKinases_with1.T.float()) # Output shape: (64,214)
        # Calculating the maximum of each row to normalize logits so that softmax doesn't overflow
        maxlogits = torch.max(logits, dim=1, keepdim=True)[0] # Output Shape: (64,1)
        # Find the class index for each data point (the class with maximum F score)
        outclassidx = torch.argmax(logits, dim=1) # Output Shape: (64)
        ## Softmax
        denom = torch.sum(torch.exp(logits - maxlogits), dim=1)
        M = torch.sum(pred * batch_CEs, dim=1) - maxlogits.squeeze()
        rightprobs = torch.exp(M) / (denom + 1e-15) # Softmax
        ## Cross Entropy
        P = torch.clamp(rightprobs, min=1e-15, max=1.1)
        loss = torch.mean(-torch.log(P))

        # Calculate Gradients and clip
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.CLIP_GRADIENTS)
        
        # Update Weights
        self.optimizer.step()

        return loss.item(), outclassidx, batch_TCIs


    def eval_step(self, val_dataset, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase, ValKinaseUniProtIDs, mlb_Val, binlabels_true_Val):
        self.model.eval()
        
        UniProtIDs, probabilities = self.predict(val_dataset.DE, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase)
        UniProtIDs = UniProtIDs[0]
        probabilities = probabilities[0]
        predlabels = [[label] for label in UniProtIDs]
        binlabels_pred = mlb_Val.transform(predlabels)
        Val_Evaluation = GetAccuracyMultiLabel(UniProtIDs, probabilities, ValKinaseUniProtIDs, val_dataset.TCI)
        print(classification_report(binlabels_true_Val, binlabels_pred, target_names=mlb_Val.classes_) + '\n\n\n' + 'Acccuracy_Val: {}  Loss_Val: {} Top5Accuracy: {} Top10Accuracy: {}'.format(Val_Evaluation["Accuracy"], Val_Evaluation["Loss"], Val_Evaluation["Top5Acc"], Val_Evaluation["Top10Acc"]))

        return Val_Evaluation, UniProtIDs, probabilities, mlb_Val

    
    def train(self, train_dataset, val_dataset, epochcount,
            ValCandidatekinaseEmbeddings=None,
            ValCandidateKE_to_Kinase=None, ValKinaseUniProtIDs=None, ValisMultiLabel=True, ValCandidateUniProtIDs=None):
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.training_epochs = epochcount
        self.num_examples = len(train_dataset)

        # Create Dataloaders
        #self.train_dataloader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE, shuffle = True)
        #val_dataloader = DataLoader(val_dataset, batch_size = config.BATCH_SIZE, shuffle = False)
        #test_dataloader = DataLoader(test_dataset, batch_size = config.BATCH_SIZE, shuffle = False)

        print("Number of Train data: {} Number of Val data: {}".format(len(train_dataset.DE), len(val_dataset.DE)))
        
        if ValKinaseUniProtIDs is not None:
            mlb_Val = MultiLabelBinarizer()
            binlabels_true_Val = mlb_Val.fit_transform(ValKinaseUniProtIDs)

        # For all epochs
        for epoch in range(self.training_epochs):
            print("===================================\nepoch: {}\t".format(self._epochs_completed))

            num_batches = int(self.num_examples/config.BATCH_SIZE) + 1
            loss_train = 0
            accuracy_train = 0

            # Run through all batches
            for _ in range(num_batches):
                
                loss, outclassidx, batch_TCIs = self.train_step(train_dataset)
                accuracy = accuracy_score(batch_TCIs, outclassidx, normalize=True)
                accuracy_train += accuracy
                loss_train += loss / num_batches
            accuracy_train /= num_batches

            print("train_loss: {:.3f}, train_acc: {:.3f}".format(loss_train, accuracy_train))

            if val_dataset.DE is not None:
                Val_Evaluation, UniProtIDs, probabilities, mlb_Val = self.eval_step(val_dataset, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase, ValKinaseUniProtIDs, mlb_Val, binlabels_true_Val)
                print("Val_loss: {:.3f}, Val_acc: {:.3f}".format(Val_Evaluation["Loss"], Val_Evaluation["Accuracy"]))
            
            self.lr_scheduler.step()

        if val_dataset.DE is not None:
            return accuracy_train, loss_train, Val_Evaluation, UniProtIDs, probabilities, mlb_Val, binlabels_true_Val
        else:
            return accuracy_train, loss_train, None, None, None, None, None


    def predict(self,  DataEmbedding, TestCandidateKinases, CandidateKE_to_Kinase):
        self.model.eval()
        
        allUniProtIDs = []
        allprobs = []

        TestCandidateKinases_with1 = torch.from_numpy(np.c_[ TestCandidateKinases, np.ones(len(TestCandidateKinases))])
        
        DataEmbedding.to(self.device)
        TestCandidateKinases_with1.to(self.device)
        # seq_len = [self.seq_lens] * len(DataEmbedding)

        with torch.no_grad():

            pred = self.model(DataEmbedding.float())

            # Calculate F = DE * W * CE for all the CEs in unique class embeddings (all the kinases)
            logits = torch.matmul(pred, TestCandidateKinases_with1.T.float())
            # Calculating the maximum of each row to normalize logits so that softmax doesn't overflow
            # maxlogits = torch.max(logits, dim=1, keepdim=True)[0]
            # Find the class index for each data point (the class with maximum F score)
            outclassidx = torch.argmax(logits, dim=1)
                    
            classes = TestCandidateKinases[outclassidx]
            # get UniProtIDs for predicted classes and return them
            UniProtIDs =[]
            for KE in classes:
                UniProtIDs.append(CandidateKE_to_Kinase[tuple(KE)])
            UniProtIDs = np.array(UniProtIDs)
            allUniProtIDs.append(UniProtIDs) # Belki silinebilir
            probabilities = softmax(logits, axis =1)
        allprobs.append(probabilities) # Bu da belki silinebilir
            
        return allUniProtIDs, allprobs

    




