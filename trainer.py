import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
import os
from sklearn.preprocessing import MultiLabelBinarizer

from dataset import CustomDataset
from model import MyModel


class Trainer:
    def __init__(self, model, criterion, optimizer, batch_size, device='cpu'):
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.99954, last_epoch=-1)
        self.scheduler_step = 0
        

        self.batch_size = batch_size
        self._start = 0
        self._epochs_completed = 0


    def train_step(self, train_dataset):
        self.model.train()

        batch_Xs, batch_CEs, batch_TCIs, self._start, self._epochs_completed = train_dataset.next_batch(self._start, self._epochs_completed, self.batch_size)

        self.optimizer.zero_grad()

        embedding = self.model(batch_Xs, batch_CEs, batch_TCIs, train_dataset.TrainCandidateKinases_with1)
        
        Matmul = torch.matmul(embedding, self.model.W)
        # Calculate F = DE * W * CE for all the CEs in unique class embeddings (all the kinases)
        logits = torch.matmul(Matmul, self.model.CKE.T)
        # Calculating the maximum of each row to normalize logits so that softmax doesn't overflow
        maxlogits = torch.max(logits, dim=1, keepdim=True)[0]
        # Find the class index for each data point (the class with maximum F score)
        outclassidx = torch.argmax(logits, dim=1)
        ## Softmax
        denom = torch.sum(torch.exp(logits - maxlogits.unsqueeze(1)), dim=1)
        M = torch.sum(Matmul * self.CE, dim=1) - maxlogits.squeeze()
        rightprobs = torch.exp(M) / (denom + 1e-15) # Softmax
        ## Cross Entropy
        P = torch.clamp(rightprobs, min=1e-15, max=1.1)
        loss = torch.mean(-torch.log(P))

        loss.backward()

        # Update Weights
        self.optimizer.step()
        self.lr_scheduler.step(self.scheduler_step)

        return loss.item(), outclassidx.item()

    def eval_step(self, batch):
        pass

    def predict_step(self, inputs):
        pass


    def train(self, train_dataset, X, ClassEmbedding, TrainCandidateKinases, TrueClassIDX, epochcount, 
            ValDE = None, ValCandidatekinaseEmbeddings=None,
            ValCandidateKE_to_Kinase=None, ValKinaseUniProtIDs=None,
            ValKinaseEmbeddings=None, ValisMultiLabel=True, 
            Val_TrueClassIDX=None, ValCandidateUniProtIDs=None):
        

        self.train_dataset = train_dataset

        print("Number of Train data: {} Number of Val data: {}".format(len(X), len(ValDE)))
        
        self.training_epochs = epochcount
        self.num_examples = len(X)

        if ValKinaseUniProtIDs is not None:
            mlb_Val = MultiLabelBinarizer()
            binlabels_true_Val = mlb_Val.fit_transform(ValKinaseUniProtIDs)
        
        Bestaccuracy_Val = 0
        Bestaccuracy_Train = 0
        Best_loss = 0


        #print("Epoch,," + ','.join(['TrainAcc_{}'.format(i) for i in range(self.Params["NumofModels"])]) + ',,' + ','.join(['ValAcc_{}'.format(i) for i in range(self.Params["NumofModels"])])+ ',,' + ','.join(['TestAcc_{}'.format(i) for i in range(self.Params["NumofModels"])]) + ',,' + 'ValAcc_Ensemble,TestAcc_Ensemble', file=Allresults)

        for epoch in range(self.training_epochs):
            print("===================================\nepoch: {}\t".format(self._epochs_completed))

            num_batches = int(self.num_examples/self.Params["batch_size"]) + 1

            ValUniProtIDs = []
            ValProbs = []
            loss_train = 0
            accuracy_train = 0
            
            for b in range(num_batches):
                self.scheduler_step += 1 ## BAK nerede kullanildigina
                loss, outclassidx = self.train_step(train_dataset)
                train_loss += loss
            train_loss /= len(train_loader)
            train_losses.append(train_loss)

    




