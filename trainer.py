import torch
import torch.optim as optim
from torch.utils.data import DataLoader

import time
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import accuracy_score, classification_report

from Utils.utils import get_eval_predictions


class Trainer:
    def __init__(self, model, kinase_model, optimizer, scheduler, args):
        self.model = model
        self.kinase_model = kinase_model
        self.optimizer = optimizer
        self.device = args.DEVICE
        self.lr_scheduler = scheduler
        self.args = args

        # Set model device
        self.model.to(self.device)
        if kinase_model is not None:
            self.kinase_model.to(self.device)
            if args.TRAIN_KINASE == False:
                self.kinase_model.eval()


    def train_step(self, train_dataloader):
        
        # Set model to training mode
        self.model.train()

        epoch_loss, epoch_accuracy = 0, 0

        for _, (phosphosite,kinase,y) in enumerate(train_dataloader):

            # Set device of data
            if self.args.HF_ONLY_ID:
                phosphosite = phosphosite.to(self.device)
            else:
                phosphosite['input_ids'] = phosphosite['input_ids'].to(self.device)
                phosphosite['attention_mask'] = phosphosite['attention_mask'].to(self.device)
            kinase = kinase.to(self.device)
            y = y.to(self.device)
            ### Train candidate kinase device datasete ekle

            # Reset Gradients
            self.optimizer.zero_grad()

            # Prediction
            y_pred = self.model(phosphosite)

            # ESM Kinase Model Pass
            if self.args.USE_ESM_KINASE:
                kinase_embeddings = self.kinase_model(kinase)
                # Calculation Loss
                loss, outclassidx = self.criterion(y_pred, kinase_embeddings)
            else:
                loss, outclassidx = self.criterion(y_pred, kinase)
            
            # Calculating Gradients
            loss.backward()
            #torch.nn.utils.clip_grad_norm_(self.model.parameters(), config.CLIP_GRADIENTS)
            
            # Update Weights
            self.optimizer.step()
            
            # Calculating Performance Metrics
            epoch_accuracy += accuracy_score(y.cpu(), outclassidx.cpu(), normalize=True)
            epoch_loss += loss.item()

        epoch_loss /= len(train_dataloader)
        epoch_accuracy /= len(train_dataloader)

        return epoch_loss, epoch_accuracy


    def eval_step(self, val_dataloader, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase, ValKinaseUniProtIDs, mlb_Val):
        
        self.model.eval()
        
        #X, CE, y = next(iter(val_dataloader))
        phosphosite, kinase, labels = val_dataloader.phosphosite, val_dataloader.kinase_with_1, val_dataloader.labels
        if self.args.USE_ESM_KINASE:
            candidate_kinase_with_1 = torch.from_numpy(np.c_[ ValCandidatekinaseEmbeddings, np.ones(len(ValCandidatekinaseEmbeddings))]).to(torch.int64)
        else:
            candidate_kinase_with_1 = torch.from_numpy(np.c_[ ValCandidatekinaseEmbeddings, np.ones(len(ValCandidatekinaseEmbeddings))]).float()
        # Set Data Device
        if self.args.HF_ONLY_ID:
            phosphosite = phosphosite.to(self.device)
        else:
            phosphosite['input_ids'] = phosphosite['input_ids'].to(self.device)
            phosphosite['position_ids'] = phosphosite['position_ids'].to(self.device)
        candidate_kinase_with_1 = candidate_kinase_with_1.to(self.device)

        with torch.no_grad():

            pred = self.model(phosphosite)
            if self.args.USE_ESM_KINASE:
                candidate_kinase_with_1 = self.kinase_model(candidate_kinase_with_1)
            
            # Equation 5 from paper
            logits = torch.matmul(pred, candidate_kinase_with_1.T)
            outclassidx = torch.argmax(logits, dim=1) 
            classes = ValCandidatekinaseEmbeddings[outclassidx.cpu()]
            probabilities = torch.nn.functional.softmax(logits, dim=1)

        probabilities = probabilities.cpu().numpy()

        # get UniProtIDs for predicted classes and return them
        UniProtIDs =[]
        for c in classes:
            UniProtIDs.append(ValCandidateKE_to_Kinase[tuple(c)])
        UniProtIDs = np.array(UniProtIDs)

        Val_Evaluation, binlabels_pred = get_eval_predictions(UniProtIDs, probabilities, ValKinaseUniProtIDs, labels, mlb_Val)

        return Val_Evaluation, UniProtIDs, probabilities, binlabels_pred
    

    def train(self,
              train_dataset,
              val_dataset,
              ValCandidatekinaseEmbeddings=None,
              ValCandidateKE_to_Kinase=None, 
              ValKinaseUniProtIDs=None):
        
        # Datasets and Dataloaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset

        train_dataloader = DataLoader(train_dataset, batch_size = self.args.BATCH_SIZE, shuffle = True)
        #if val_dataset is not None:
            #val_dataloader = DataLoader(val_dataset, batch_size = len(val_dataset), shuffle = False)

        # Set it for loss calculation
        self.train_dataset.kinase_set_with_1 = self.train_dataset.kinase_set_with_1.to(self.device)

        # For Classification Report
        if ValKinaseUniProtIDs is not None:
            mlb_Val = MultiLabelBinarizer()
            binlabels_true_Val = mlb_Val.fit_transform(ValKinaseUniProtIDs)

        
        ### TRAINING STARTS ###
        epochs_start_time = time.time()

        for epoch in range(self.args.NUM_EPOCHS):
            print("===================================\nepoch: {}\t".format(epoch))
            train_step_start_time = time.time()

            train_loss, train_acc = self.train_step(train_dataloader)

            print(f'Train Step takes {time.time() - train_step_start_time} seconds')

            if val_dataset is not None:
                Val_Evaluation, UniProtIDs, probabilities, binlabels_pred = self.eval_step(val_dataset, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase, ValKinaseUniProtIDs, mlb_Val)
            
            # Change learning rate with scheduler
            self.lr_scheduler.step()

            # Epoch results (Train)
            print("train_loss: {:.3f}, train_acc: {:.3f}".format(train_loss, train_acc))
            
            # Epoch results (Validation)
            if val_dataset is not None:
                #print(classification_report(binlabels_true_Val, binlabels_pred, target_names=mlb_Val.classes_) + '\n\n')
                print('Acccuracy_Val: {}  Loss_Val: {} Top5Accuracy: {} Top10Accuracy: {}'\
                      .format(Val_Evaluation["Accuracy"], Val_Evaluation["Loss"], Val_Evaluation["Top5Acc"], Val_Evaluation["Top10Acc"]))

        print(f'Epochs time for 1 model: {time.time() - epochs_start_time} seconds')

        if val_dataset is not None:
            return train_acc, train_loss, Val_Evaluation, UniProtIDs, probabilities, mlb_Val, binlabels_true_Val
        else:
            return train_acc, train_loss, None, None, None, None, None 


    def criterion(self, y_pred, kinase):
        
        # F: Compatibility Function
        # Calculate F = DE * W * CE (I think here is CKE) for all the CEs in unique class embeddings (all the kinases)        
        # y_pred = DE * W
        # Equation 3 from paper
        logits = torch.matmul(y_pred, self.train_dataset.kinase_set_with_1.T) # Output shape: (64,214)
        
        # Calculating the maximum of each row to normalize logits so that softmax doesn't overflow
        maxlogits = torch.max(logits, dim=1, keepdim=True)[0] # Output Shape: (64,1)
        
        ## p(y|x): Equation 1 from paper
        numerator = torch.sum(y_pred * kinase, dim=1) - maxlogits.squeeze()
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
