import torch
import numpy as np
import time
from trainer import Trainer
from model import  Bi_RNN
from utils import ensemble, get_eval_predictions
from sklearn.metrics import classification_report
from create_dataset import create_datasets
import config as config

import warnings
warnings.filterwarnings("ignore")


# Create Datasets
create_train_data_time = time.time()
train_dataset, phosphosite_seq_size, embed_scaler = create_datasets(config.TRAIN_DATA, mode='train')
print(f'Creating train data takes: {time.time() - create_train_data_time}')
val_dataset, _, KE, val_candidate_kinase_embeddings, val_candidate_ke_to_kinase, val_kinase_uniprotid, val_candidate_uniprotid = create_datasets(config.VAL_DATA, config.VAL_KINASE_CANDIDATES, mode='val',embed_scaler=embed_scaler)
test_dataset, _, KE, test_candidate_kinase_embeddings, test_candidate_ke_to_kinase, test_kinase_uniprotid, test_candidate_uniprotid = create_datasets(config.TEST_DATA, config.TEST_KINASE_CANDIDATES, mode='test',embed_scaler=embed_scaler)


# Define Models

models = [
    Bi_RNN(vocabnum = phosphosite_seq_size[1], seq_lens = phosphosite_seq_size[0], ClassEmbeddingsize = KE.Embedding_size)
    for _ in range(config.NUM_OF_MODELS)
]

# Define Optimizers
optimizers = [
    torch.optim.Adam(models[i].parameters(), lr=config.LEARNING_RATE)
    for i in range(config.NUM_OF_MODELS)
]

# Define Trainers
trainers = [
    Trainer(models[i], optimizers[i], device=config.DEVICE)
    for i in range(config.NUM_OF_MODELS)
]

# Train Eval Lists
AllAccuracyTrains = np.zeros(config.NUM_OF_MODELS)
AllAccuracyLoss = np.zeros(config.NUM_OF_MODELS)
AllAccuracyVals = np.zeros(config.NUM_OF_MODELS)
AllAccuracyValProbs = [] # np.empty(config.NUM_OF_MODELS)


# Train the models
for i, model in enumerate(models):
    accuracy_train, loss_train, Val_Evaluation, ValUniProtIDs, ValProbs, mlb_val, binlabels_true_Val  = trainers[i].train(train_dataset, val_dataset, num_epochs=1, ValCandidatekinaseEmbeddings=val_candidate_kinase_embeddings, ValCandidateKE_to_Kinase=val_candidate_ke_to_kinase, ValKinaseUniProtIDs=val_kinase_uniprotid)
    
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

# Results (Test)
#if config.TEST_DATA != '':
#    for trainer in trainers:
#       TestUniProtIDs, test_probabilities = trainer.predict(test_dataset.DE, test_candidate_kinase_embeddings, test_candidate_ke_to_kinase)
#        TestUniProtIDs, test_probabilities = ensemble(TestUniProtIDs, test_probabilities, test_candidate_uniprotid)
#        Test_Evaluation = GetAccuracyMultiLabel(TestUniProtIDs, test_probabilities, test_kinase_uniprotid, test_dataset.TCI)
#        print('Acccuracy_Test: {}  Loss_Test: {} Top3Accuracy: {} Top5Accuracy: {} Top10Accuracy: {}'.format(Test_Evaluation["Accuracy"], Test_Evaluation["Loss"], Test_Evaluation["Top3Acc"], Test_Evaluation["Top5Acc"], Test_Evaluation["Top10Acc"]))
        