import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

from utils import get_eval_predictions
import config as config


class Trainer:
    def __init__(self, model, optimizer, device='cpu'):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.99954, last_epoch=-1)

        # Set model device
        self.model.to(self.device)


    def train_step(self, train_dataloader):
        
        # Set model to training mode
        self.model.train()

        epoch_loss, epoch_accuracy = 0, 0

        for _, (X,CE,y) in enumerate(train_dataloader):
            X = X.to(self.device)
            CE = CE.to(self.device)
            y = y.to(self.device)
            ### Train candidate kinase device datasete ekle

            # Reset Gradients
            self.optimizer.zero_grad()

            # Prediction
            y_pred = self.model(X) # 0.22sn

            # Calculation Loss
            loss, outclassidx = self.criterion(y_pred, CE)
            
            # Calculating Gradients
            loss.backward() # 0.44 sn
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.CLIP_GRADIENTS)
            
            # Update Weights
            self.optimizer.step()
            
            # Calculating Performance Metrics
            epoch_accuracy += accuracy_score(y, outclassidx, normalize=True)
            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)
        epoch_accuracy /= len(train_dataloader)

        return epoch_loss, epoch_accuracy


    def eval_step(self, val_dataloader, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase, ValKinaseUniProtIDs, mlb_Val):
        
        self.model.eval()
        
        #X, CE, y = next(iter(val_dataloader))
        X, _, y = val_dataloader.DE, val_dataloader.ClassEmbedding_with1, val_dataloader.TCI
        
        TestCandidateKinases_with1 = torch.from_numpy(np.c_[ ValCandidatekinaseEmbeddings, np.ones(len(ValCandidatekinaseEmbeddings))]).float()
        
        X.to(self.device)
        TestCandidateKinases_with1.to(self.device)

        #allUniProtIDs = []
        #allprobs = []

        with torch.no_grad():

            pred = self.model(X)
            
            # Equation 5 from paper
            logits = torch.matmul(pred, TestCandidateKinases_with1.T)
            outclassidx = torch.argmax(logits, dim=1) 
            classes = ValCandidatekinaseEmbeddings[outclassidx]
            probabilities = torch.nn.functional.softmax(logits, dim=1)

        # get UniProtIDs for predicted classes and return them
        UniProtIDs =[]
        for c in classes:
            UniProtIDs.append(ValCandidateKE_to_Kinase[tuple(c)])
        UniProtIDs = np.array(UniProtIDs)

        Val_Evaluation, binlabels_pred = get_eval_predictions(UniProtIDs, probabilities, ValKinaseUniProtIDs, y, mlb_Val)

        return Val_Evaluation, UniProtIDs, probabilities, binlabels_pred
    

    def train(self,
               train_dataset,
               val_dataset, 
               num_epochs,
               ValCandidatekinaseEmbeddings=None,
               ValCandidateKE_to_Kinase=None, 
               ValKinaseUniProtIDs=None):
        
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        train_dataloader = DataLoader(train_dataset, batch_size = config.BATCH_SIZE, shuffle = True)
        #if val_dataset is not None:
            #val_dataloader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = False)

        if ValKinaseUniProtIDs is not None:
            mlb_Val = MultiLabelBinarizer()
            binlabels_true_Val = mlb_Val.fit_transform(ValKinaseUniProtIDs)

        epochs_start_time = time.time()

        for epoch in range(num_epochs):
            print("===================================\nepoch: {}\t".format(epoch))
            
            train_step_start_time = time.time()
            train_loss, train_acc = self.train_step(train_dataloader)
            print(f'Train Step takes {time.time() - train_step_start_time} seconds')

            if val_dataset is not None:
                Val_Evaluation, UniProtIDs, probabilities, binlabels_pred = self.eval_step(val_dataset, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase, ValKinaseUniProtIDs, mlb_Val)
            
            self.lr_scheduler.step()

            # Epoch results
            print("train_loss: {:.3f}, train_acc: {:.3f}".format(train_loss, train_acc))
            
            if val_dataset is not None:
                print(classification_report(binlabels_true_Val, binlabels_pred, target_names=mlb_Val.classes_) + '\n\n')
                print('Acccuracy_Val: {}  Loss_Val: {} Top5Accuracy: {} Top10Accuracy: {}'.format(Val_Evaluation["Accuracy"], Val_Evaluation["Loss"], Val_Evaluation["Top5Acc"], Val_Evaluation["Top10Acc"]))


        print(f'Epochs time for 1 model: {time.time() - epochs_start_time} seconds')

        if val_dataset is not None:
            return train_acc, train_loss, Val_Evaluation, UniProtIDs, probabilities, mlb_Val, binlabels_true_Val
        else:
            return train_acc, train_loss, None, None, None, None, None 


    def criterion(self, y_pred, CE):
        
        # F: Compatibility Function
        # Calculate F = DE * W * CE (I think here is CKE) for all the CEs in unique class embeddings (all the kinases)        
        # y_pred = DE * W
        # Equation 3 from paper
        logits = torch.matmul(y_pred, self.train_dataset.TrainCandidateKinases_with1.T) # Output shape: (64,214)
        # Calculating the maximum of each row to normalize logits so that softmax doesn't overflow
        maxlogits = torch.max(logits, dim=1, keepdim=True)[0] # Output Shape: (64,1)
        
        ## p(y|x): Equation 1 from paper
        numerator = torch.sum(y_pred * CE, dim=1) - maxlogits.squeeze()
        denominator = torch.sum(torch.exp(logits - maxlogits), dim=1)
        softmax_out = torch.exp(numerator) / (denominator + 1e-15)
        
        ## Cross Entropy
        # Equation 4 from paper
        P = torch.clamp(softmax_out, min=1e-15, max=1.1)
        loss = torch.mean(-torch.log(P))

        # Find the class index for each data point (the class with maximum F score)
        # I detached it due to preventing unnecessary gradient calculation
        outclassidx = torch.argmax(logits.detach(), dim=1) # Output Shape: (64)

        return loss, outclassidx

    



