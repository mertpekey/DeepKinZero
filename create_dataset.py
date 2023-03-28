from dataset import CustomDataset
from data.sequence_data import all_seq_dataset

from utils import FindTrueClassIndices
from data.kinase_embeddings import KinaseEmbedding

from sklearn import preprocessing
import config as config


def create_datasets():
    # Define Dataset
    KE = KinaseEmbedding(Family = True, Group = True, Pathway = False, Kin2Vec=True, Enzymes = True)
    TrainData = config.TRAIN_DATA
    TestData = config.TEST_DATA
    ValData = config.VAL_DATA
    TestKinaseCandidates = config.TEST_KINASE_CANDIDATES
    ValKinaseCandidates = config.VAL_KINASE_CANDIDATES
    TestisLabeled = config.TEST_IS_LABELED
    train_dataset, val_dataset, test_dataset = None, None, None
    NormalizeDE = True

    ### Train Data ###
    if TrainData != '':
        # Create the training dataset
        TrainDS = all_seq_dataset()
        TrainDS.get_data(TrainData, KE, is_labeled=True)
        
        ### Model Input (Train) ### (12901,13,100)
        TrainSeqEmbedded = TrainDS.get_embedded_seqs(AminoAcidProperties=False, ProtVec=True) # Get the sequence embeddings
        #Normalize Training data
        if NormalizeDE:
            TrainSeqEmbeddedreshaped = TrainSeqEmbedded.reshape(TrainSeqEmbedded.shape[0], TrainSeqEmbedded.shape[1] * TrainSeqEmbedded.shape[2])
            SeqEmbedScaler = preprocessing.StandardScaler().fit(TrainSeqEmbeddedreshaped)
            TrainSeqEmbeddedreshaped = SeqEmbedScaler.transform(TrainSeqEmbeddedreshaped)
            TrainSeqEmbedded = TrainSeqEmbeddedreshaped.reshape(TrainSeqEmbedded.shape[0], TrainSeqEmbedded.shape[1], TrainSeqEmbedded.shape[2])
        TrueClassIDX = FindTrueClassIndices(TrainDS.KinaseEmbeddings, TrainDS.UniqueKinaseEmbeddings) # KinaseEmb (12901,727), Unique (214,727), TCI (12901)
        #### FINAL TRAIN DATASET ###
        train_dataset = CustomDataset(TrainSeqEmbedded, TrainDS.KinaseEmbeddings, TrueClassIDX, TrainDS.UniqueKinaseEmbeddings, is_train=True, FakeRand=False, shuffle=True)

    ### Val Data ###
    if ValData != '':
        ValDS = all_seq_dataset()
        ValDS.get_data(ValData, KE, is_labeled=True, MultiLabel=True)

        ### Model Input (Val) ### (80, 13, 100)
        ValSeqEmbedded = ValDS.get_embedded_seqs(AminoAcidProperties=False, ProtVec=True)
        if NormalizeDE:
            ValSeqEmbeddedreshaped = ValSeqEmbedded.reshape(ValSeqEmbedded.shape[0], ValSeqEmbedded.shape[1] * ValSeqEmbedded.shape[2])
            ValSeqEmbeddedreshaped = SeqEmbedScaler.transform(ValSeqEmbeddedreshaped)
            ValSeqEmbedded = ValSeqEmbeddedreshaped.reshape(ValSeqEmbedded.shape[0], ValSeqEmbedded.shape[1], ValSeqEmbedded.shape[2])
        
        # (17,), (17,727), (17,), (17,), (17,)
        if ValKinaseCandidates != '':
            ValCandidatekinases, ValCandidatekinaseEmbeddings, ValCandidateindices, ValCandidateKE_to_Kinase, ValCandidate_UniProtIDs = KE.read_kinases_from_path(ValKinaseCandidates)
            Val_TrueClassIDX = FindTrueClassIndices(ValDS.KinaseEmbeddings, ValCandidatekinaseEmbeddings, True) # (80,), icindekiler 1 elemanlik list
            #### FINAL VAL DATASET ###
            val_dataset = CustomDataset(ValSeqEmbedded, ValDS.KinaseEmbeddings, Val_TrueClassIDX, ValDS.UniqueKinaseEmbeddings, is_train=False, FakeRand=False, shuffle=False)


    ### Test Data ###
    if TestData != '':
        TestDS = all_seq_dataset()
        TestDS.getdata(TestData, KE, islabeled=TestisLabeled, MultiLabel=True)
        ### Model Input (Test) ###
        TestSeqEmbedded = TestDS.get_embedded_seqs(AminoAcidProperties=False, ProtVec=True)
        if NormalizeDE:
            TestSeqEmbeddedreshaped = TestSeqEmbedded.reshape(TestSeqEmbedded.shape[0], TestSeqEmbedded.shape[1] * TestSeqEmbedded.shape[2])
            TestSeqEmbeddedreshaped = SeqEmbedScaler.transform(TestSeqEmbeddedreshaped)
            TestSeqEmbedded = TestSeqEmbeddedreshaped.reshape(TestSeqEmbedded.shape[0], TestSeqEmbedded.shape[1], TestSeqEmbedded.shape[2])
        
        if TestKinaseCandidates != '':
            Candidatekinases, CandidatekinaseEmbeddings, Candidateindices, CandidateKE_to_Kinase, Candidate_UniProtIDs = KE.readKinases(TestKinaseCandidates)
            if TestisLabeled:
                Test_TrueClassIDX = FindTrueClassIndices(TestDS.KinaseEmbeddings, CandidatekinaseEmbeddings, True)
                #### FINAL TEST DATASET ###
                test_dataset = CustomDataset(TestSeqEmbedded, TestDS.KinaseEmbeddings, Test_TrueClassIDX, TestDS.UniqueKinaseEmbeddings, is_train=False, FakeRand=False, shuffle=False)
    

    Phosphosite_Seq_Size = all_seq_dataset.Get_SeqSize(AminoAcidProperties=False, ProtVec=True)

    return train_dataset, val_dataset, test_dataset, Phosphosite_Seq_Size, KE, ValCandidatekinaseEmbeddings, ValCandidateKE_to_Kinase, ValDS.KinaseUniProtIDs, ValCandidate_UniProtIDs