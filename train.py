import torch
import torch.optim as optim
import numpy as np
from trainer import Trainer
from model import  Bi_LSTM, HuggingFace_Transformer, Transformer_LSTM
from utils import ensemble, get_eval_predictions
from sklearn.metrics import classification_report
from create_dataset import create_datasets
import config as config
import os
import sys


import warnings
warnings.filterwarnings("ignore")

def train_model(args):

    # Create Datasets

    train_dataset, phosphosite_seq_size, embed_scaler = create_datasets(args.TRAIN_DATA, 
                                                                        mode='train', 
                                                                        is_huggingface=args.IS_HUGGINGFACE)
    val_dataset, _, KE, val_candidate_kinase_embeddings, \
    val_candidate_ke_to_kinase, val_kinase_uniprotid, \
    val_candidate_uniprotid = create_datasets(args.VAL_DATA, 
                                              args.VAL_KINASE_CANDIDATES, 
                                              mode='val',
                                              embed_scaler=embed_scaler, 
                                              is_huggingface=args.IS_HUGGINGFACE)

    # Define Model and Trainers
    models, optimizers, schedulers, trainers = [], [], [], []
    
    for i in range(args.NUM_OF_MODELS):
        if args.MODEL_TYPE == 'BiLSTM':
            trainer_model = Bi_LSTM(vocabnum = phosphosite_seq_size[1], seq_lens = phosphosite_seq_size[0], ClassEmbeddingsize = KE.Embedding_size)
        elif args.MODEL_TYPE == 'ProtBERT':
            trainer_model = HuggingFace_Transformer(hf_checkpoint="Rostlab/prot_bert", ClassEmbeddingsize = KE.Embedding_size)
        elif args.MODEL_TYPE == 'Transformer_LSTM':
            trainer_model = Transformer_LSTM(vocabnum = phosphosite_seq_size[1], 
                                             seq_lens = phosphosite_seq_size[0], 
                                             ClassEmbeddingsize = KE.Embedding_size, 
                                             hf_checkpoint="Rostlab/prot_bert")
        elif args.MODEL_TYPE == 'ProtT5':
            pass
        else:
            print('Input valid model name')
            sys.exit()
        
        trainer_optimizer = torch.optim.Adam(trainer_model.parameters(), lr=args.LEARNING_RATE)
        trainer_scheduler = optim.lr_scheduler.ExponentialLR(trainer_optimizer, gamma=0.99954, last_epoch=-1)

        models.append(trainer_model)
        optimizers.append(trainer_optimizer)
        schedulers.append(trainer_scheduler)
        trainers.append(Trainer(models[i], optimizers[i], scheduler=schedulers[i], device=args.DEVICE))
    

    # Train Eval Lists
    AllAccuracyTrains = np.zeros(args.NUM_OF_MODELS)
    AllAccuracyLoss = np.zeros(args.NUM_OF_MODELS)
    AllAccuracyVals = np.zeros(args.NUM_OF_MODELS)
    AllAccuracyValProbs = [] # np.empty(config.NUM_OF_MODELS)

    # Train the models
    for i, model in enumerate(models):
        accuracy_train, loss_train, Val_Evaluation, ValUniProtIDs, ValProbs, mlb_val, binlabels_true_Val  = trainers[i].train(train_dataset, 
                                                                                                                              val_dataset, 
                                                                                                                              num_epochs=args.NUM_EPOCHS, 
                                                                                                                              ValCandidatekinaseEmbeddings=val_candidate_kinase_embeddings, 
                                                                                                                              ValCandidateKE_to_Kinase=val_candidate_ke_to_kinase, 
                                                                                                                              ValKinaseUniProtIDs=val_kinase_uniprotid)
        
        AllAccuracyTrains[i] = accuracy_train
        AllAccuracyLoss[i] = loss_train
        AllAccuracyVals[i] = Val_Evaluation["Accuracy"]
        AllAccuracyValProbs.append(ValProbs) # AllAccuracyValProbs[i] = ValProbs

        if args.SAVE_MODEL:
            save_filepath = os.path.join('pretrained_files','pretrained_models',args.MODEL_TYPE, args.SAVE_FILEPATH + f'_{i}')
            state = {
                'state_dict': trainers[i].model.state_dict(),
                'optimizer': trainers[i].optimizer.state_dict()
            }
            torch.save(state, save_filepath)

    # Ensemble the results (Train)
    accuracy_train_ensemble = np.mean(AllAccuracyTrains)
    loss_train_ensemble = np.mean(AllAccuracyLoss)

    print("Ensembled Train_Loss: {:.3f}, Ensembled Train_Acc: {:.3f}".format(loss_train_ensemble, accuracy_train_ensemble))

    # Ensemble the results (Validation)
    if Val_Evaluation is not None:
        ValUniProtIDs, Valprobabilities = ensemble(ValUniProtIDs, AllAccuracyValProbs, val_candidate_uniprotid)
        Val_Evaluation, binlabels_pred = get_eval_predictions(ValUniProtIDs, Valprobabilities, val_kinase_uniprotid, val_dataset.TCI, mlb_val)
        print(classification_report(binlabels_true_Val, binlabels_pred, target_names=mlb_val.classes_) + \
              '\n\n\n' + 'Val_Acc: {}  Val_Loss: {} Val_Acc_Top3: {} Val_Acc_Top5: {} Val_Acc_Top10: {}'\
                .format(Val_Evaluation["Accuracy"], Val_Evaluation["Loss"], Val_Evaluation["Top3Acc"], Val_Evaluation["Top5Acc"], Val_Evaluation["Top10Acc"]))
