import torch
import numpy as np
from trainer import Trainer
from dataset import CustomDataset
from data.sequence_data import all_seq_dataset
from model import  MyModel

from train_utils import ensemble, GetAccuracyMultiLabel
from utils import FindTrueClassIndices
from data.kinase_embeddings import KinaseEmbedding

from sklearn.metrics import classification_report
from sklearn import preprocessing


def create_datasets():
    # Define Dataset
    KE = KinaseEmbedding(Family = True, Group = True, Pathways = False, Kin2Vec=True, Enzymes = True)

    TrainData = 'Deneme'
    TestData = 'Deneme'
    ValData = 'Deneme'
    TestKinaseCandidates = 'Deneme'
    ValKinaseCandidates = 'Deneme'
    TestisLabeled = 'Deneme'

    train_dataset, val_dataset, test_dataset = None, None, None

    NormalizeDE = True
    if TrainData != '':
        # Create the training dataset
        TrainDS = all_seq_dataset()
        TrainDS.getdata(TrainData, KE, islabeled=True)
        
        # Train Input
        TrainSeqEmbedded = TrainDS.get_embedded_seqs(AminoAcidProperties=False, ProtVec=True) # Get the sequence embeddings
        #Normalize Training data
        if NormalizeDE:
            TrainSeqEmbeddedreshaped = TrainSeqEmbedded.reshape(TrainSeqEmbedded.shape[0], TrainSeqEmbedded.shape[1] * TrainSeqEmbedded.shape[2])
            SeqEmbedScaler = preprocessing.StandardScaler().fit(TrainSeqEmbeddedreshaped)
            TrainSeqEmbeddedreshaped = SeqEmbedScaler.transform(TrainSeqEmbeddedreshaped)
            TrainSeqEmbedded = TrainSeqEmbeddedreshaped.reshape(TrainSeqEmbedded.shape[0], TrainSeqEmbedded.shape[1], TrainSeqEmbedded.shape[2])
        TrueClassIDX = FindTrueClassIndices(TrainDS.KinaseEmbeddings, TrainDS.UniqueKinaseEmbeddings)
        # TRAIN DATA
        train_dataset = CustomDataset(TrainSeqEmbedded, TrainDS.KinaseEmbeddings, TrueClassIDX, TrainDS.UniqueKinaseEmbeddings, is_train=True, FakeRand=False, shuffle=True)

    if TestData != '':
        TestDS = all_seq_dataset()
        TestDS.getdata(TestData, KE, islabeled=TestisLabeled, MultiLabel=True)
        # Test Input
        TestSeqEmbedded = TestDS.Get_Embedded_Seqs(AminoAcidProperties=False, ProtVec=True)
        if NormalizeDE:
            TestSeqEmbeddedreshaped = TestSeqEmbedded.reshape(TestSeqEmbedded.shape[0], TestSeqEmbedded.shape[1] * TestSeqEmbedded.shape[2])
            TestSeqEmbeddedreshaped = SeqEmbedScaler.transform(TestSeqEmbeddedreshaped)
            TestSeqEmbedded = TestSeqEmbeddedreshaped.reshape(TestSeqEmbedded.shape[0], TestSeqEmbedded.shape[1], TestSeqEmbedded.shape[2])

    if ValData != '':
        ValDS = all_seq_dataset()
        ValDS.getdata(ValData, KE, islabeled=True, MultiLabel=True)
        # Val Input
        ValSeqEmbedded = ValDS.Get_Embedded_Seqs(AminoAcidProperties=False, ProtVec=True)
        if NormalizeDE:
            ValSeqEmbeddedreshaped = ValSeqEmbedded.reshape(ValSeqEmbedded.shape[0], ValSeqEmbedded.shape[1] * ValSeqEmbedded.shape[2])
            ValSeqEmbeddedreshaped = SeqEmbedScaler.transform(ValSeqEmbeddedreshaped)
            ValSeqEmbedded = ValSeqEmbeddedreshaped.reshape(ValSeqEmbedded.shape[0], ValSeqEmbedded.shape[1], ValSeqEmbedded.shape[2])
    
    if TestKinaseCandidates != '':
        Candidatekinases, CandidatekinaseEmbeddings, Candidateindices, CandidateKE_to_Kinase, Candidate_UniProtIDs = KE.readKinases(TestKinaseCandidates)
        if TestisLabeled:
            Test_TrueClassIDX = FindTrueClassIndices(TestDS.KinaseEmbeddings, CandidatekinaseEmbeddings, True)
            # TEST DATA
            test_dataset = CustomDataset(TestSeqEmbedded, TestDS.KinaseEmbeddings, Test_TrueClassIDX, TestDS.UniqueKinaseEmbeddings, is_train=False, FakeRand=False, shuffle=False)
    if ValKinaseCandidates != '':
        ValCandidatekinases, ValCandidatekinaseEmbeddings, ValCandidateindices, ValCandidateKE_to_Kinase, ValCandidate_UniProtIDs = KE.readKinases(ValKinaseCandidates)
        Val_TrueClassIDX = FindTrueClassIndices(ValDS.KinaseEmbeddings, ValCandidatekinaseEmbeddings, True)
        # VAL DATA
        val_dataset = CustomDataset(ValSeqEmbedded, ValDS.KinaseEmbeddings, Val_TrueClassIDX, ValDS.UniqueKinaseEmbeddings, is_train=False, FakeRand=False, shuffle=False)

    Phosphosite_Seq_Size = all_seq_dataset.Get_SeqSize(AminoAcidProperties=False, ProtVec=True)

    return train_dataset, val_dataset, test_dataset, Phosphosite_Seq_Size, KE, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase, ValDS.KinaseUniProtIDs, ValCandidate_UniProtIDs