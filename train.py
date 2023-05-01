import torch
import numpy as np
from trainer import Trainer
from model import  Bi_LSTM
from utils import ensemble, get_eval_predictions
from sklearn.metrics import classification_report
from create_dataset import create_datasets
import config as config


import warnings
warnings.filterwarnings("ignore")

def train_model(args):

    # Create Datasets

    train_dataset, phosphosite_seq_size, embed_scaler = create_datasets(args.TRAIN_DATA, mode='train')
    val_dataset, _, KE, val_candidate_kinase_embeddings, val_candidate_ke_to_kinase, val_kinase_uniprotid, val_candidate_uniprotid = create_datasets(args.VAL_DATA, args.VAL_KINASE_CANDIDATES, mode='val',embed_scaler=embed_scaler)

    # Define Model and Trainers
    models, optimizers, trainers = [], [], []
    
    for i in range(args.NUM_OF_MODELS):
        models.append(Bi_LSTM(vocabnum = phosphosite_seq_size[1], seq_lens = phosphosite_seq_size[0], ClassEmbeddingsize = KE.Embedding_size))
        optimizers.append(torch.optim.Adam(models[i].parameters(), lr=args.LEARNING_RATE))
        trainers.append(Trainer(models[i], optimizers[i], device=config.DEVICE))
    

    # Train Eval Lists
    AllAccuracyTrains = np.zeros(args.NUM_OF_MODELS)
    AllAccuracyLoss = np.zeros(args.NUM_OF_MODELS)
    AllAccuracyVals = np.zeros(args.NUM_OF_MODELS)
    AllAccuracyValProbs = [] # np.empty(config.NUM_OF_MODELS)

    # Train the models
    for i, model in enumerate(models):
        accuracy_train, loss_train, Val_Evaluation, ValUniProtIDs, ValProbs, mlb_val, binlabels_true_Val  = trainers[i].train(train_dataset, val_dataset, num_epochs=args.NUM_EPOCHS, ValCandidatekinaseEmbeddings=val_candidate_kinase_embeddings, ValCandidateKE_to_Kinase=val_candidate_ke_to_kinase, ValKinaseUniProtIDs=val_kinase_uniprotid)
        
        AllAccuracyTrains[i] = accuracy_train
        AllAccuracyLoss[i] = loss_train
        AllAccuracyVals[i] = Val_Evaluation["Accuracy"]
        AllAccuracyValProbs.append(ValProbs) # AllAccuracyValProbs[i] = ValProbs

    # Ensemble the results (Train)
    accuracy_train_ensemble = np.mean(AllAccuracyTrains)
    loss_train_ensemble = np.mean(AllAccuracyLoss)

    print("Ensemble Train_Loss: {:.3f}, Ensemble_Train_Acc: {:.3f}".format(loss_train_ensemble, accuracy_train_ensemble))

    # Ensemble the results (Validation)
    if Val_Evaluation is not None:
        ValUniProtIDs, Valprobabilities = ensemble(ValUniProtIDs, AllAccuracyValProbs, val_candidate_uniprotid)
        Val_Evaluation, binlabels_pred = get_eval_predictions(ValUniProtIDs, Valprobabilities, val_kinase_uniprotid, val_dataset.TCI, mlb_val)
        print(classification_report(binlabels_true_Val, binlabels_pred, target_names=mlb_val.classes_) + '\n\n\n' + 'Acccuracy_Val: {}  Loss_Val: {} Top3Accuracy: {} Top5Accuracy: {} Top10Accuracy: {}'.format(Val_Evaluation["Accuracy"], Val_Evaluation["Loss"], Val_Evaluation["Top3Acc"], Val_Evaluation["Top5Acc"], Val_Evaluation["Top10Acc"]))
