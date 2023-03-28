import torch
import numpy as np
from trainer import Trainer
from model import  MyModel
from utils import ensemble, GetAccuracyMultiLabel
from sklearn.metrics import classification_report
from create_dataset import create_datasets
import config as config


# Create Datasets
train_dataset, val_dataset, test_dataset, \
Phosphosite_Seq_Size, KE, ValCandidatekinaseEmbeddings, \
ValCandidateKE_to_Kinase, KinaseUniProtIDs, ValCandidate_UniProtIDs = create_datasets()

# Define Models

models = [
    MyModel(vocabnum = Phosphosite_Seq_Size[1], seq_lens = Phosphosite_Seq_Size[0], ClassEmbeddingsize = KE.Embedding_size)
    for _ in range(config.NUM_OF_MODELS)
]

# Define Optimizers
optimizers = [
    torch.optim.Adam(models[i].parameters(), lr=config.LEARNING_RATE)
    for i in range(config.NUM_OF_MODELS)
]

# Define Trainers
trainers = [
    Trainer(models[i], optimizers[i], device='cpu')
    for i in range(config.NUM_OF_MODELS)
]

# Train Eval Lists
AllAccuracyTrains = np.zeros(config.NUM_OF_MODELS)
AllAccuracyLoss = np.zeros(config.NUM_OF_MODELS)
AllAccuracyVals = np.zeros(config.NUM_OF_MODELS)
AllAccuracyValProbs = [] # np.empty(config.NUM_OF_MODELS)


# Train the models
for i, model in enumerate(models):
    accuracy_train, loss_train, Val_Evaluation, ValUniProtIDs, ValProbs, mlb_val, binlabels_true_Val  = trainers[i].train(train_dataset, val_dataset, 1, ValCandidatekinaseEmbeddings=ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase=ValCandidateKE_to_Kinase, ValKinaseUniProtIDs=KinaseUniProtIDs, ValCandidateUniProtIDs=ValCandidate_UniProtIDs)
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
    ValUniProtIDs, Valprobabilities = ensemble(ValUniProtIDs, AllAccuracyValProbs, ValCandidate_UniProtIDs)
    Val_Evaluation = GetAccuracyMultiLabel(ValUniProtIDs, Valprobabilities, KinaseUniProtIDs, val_dataset.TCI)
    predlabels = [[label] for label in ValUniProtIDs]
    binlabels_pred = mlb_val.transform(predlabels)
    print(classification_report(binlabels_true_Val, binlabels_pred, target_names=mlb_val.classes_) + '\n\n\n' + 'Acccuracy_Val: {}  Loss_Val: {} Top3Accuracy: {} Top5Accuracy: {} Top10Accuracy: {}'.format(Val_Evaluation["Accuracy"], Val_Evaluation["Loss"], Val_Evaluation["Top3Acc"], Val_Evaluation["Top5Acc"], Val_Evaluation["Top10Acc"]))

# if config.TEST_IS_LABELED:
#     print("TrainLoss = ", loss_train_ensemble, "ValLoss = ", Val_Evaluation["Loss"], "TestLoss = ", Test_Evaluations["Loss"])
#     print("TrainAccuracy = ", accuracy_train_ensemble, "Valaccuracy = ", Val_Evaluation["Accuracy"], "TestAccuracy = ", Test_Evaluations["Accuracy"])

# if config.TEST_DATA != '':
#     for trainer in trainers:
#     UniProtIDs, probabilities = trainer.predict(TestSeqEmbedded, CandidatekinaseEmbeddings, CandidateKE_to_Kinase)
#     UniProtIDs, probabilities = ensemble(UniProtIDs, probabilities, Candidate_UniProtIDs)
#     Write_predictions(OutPath, probabilities, TestDS.Sub_IDs, TestDS.Sequences, TestDS.Residues, Candidatekinases, Candidate_UniProtIDs, top_n=Top_n, sub_ID_has_original_rows=False)