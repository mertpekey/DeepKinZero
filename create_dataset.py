from dataset import DKZ_Dataset
from data.sequence_data import all_seq_dataset

from utils import FindTrueClassIndices
from data.kinase_embeddings import KinaseEmbedding

from sklearn import preprocessing
import config as config


def create_datasets(data_path, candidate_path = None, mode='train', is_labeled=True, normalize_embedding = True, embed_scaler=None):
    # Define Dataset
    KE = KinaseEmbedding(Family = True, Group = True, Pathway = False, Kin2Vec=True, Enzymes = True)

    if mode == 'train':
        # Create the training dataset
        train_ds = all_seq_dataset()
        train_ds.get_data(data_path, KE, is_labeled=is_labeled)
        
        ### Model Input (Train) ### (12901,13,100)
        phospho_embedded = train_ds.get_embedded_seqs(AminoAcidProperties=False, ProtVec=True) # Get the sequence embeddings
        
        #Normalize Training data
        if normalize_embedding:
            phospho_embedded_reshape = phospho_embedded.reshape(phospho_embedded.shape[0], phospho_embedded.shape[1] * phospho_embedded.shape[2])
            embed_scaler = preprocessing.StandardScaler().fit(phospho_embedded_reshape)
            phospho_embedded_reshape = embed_scaler.transform(phospho_embedded_reshape)
            phospho_embedded = phospho_embedded_reshape.reshape(phospho_embedded.shape[0], phospho_embedded.shape[1], phospho_embedded.shape[2])
        labels = FindTrueClassIndices(train_ds.KinaseEmbeddings, train_ds.UniqueKinaseEmbeddings) # KinaseEmb (12901,727), Unique (214,727), TCI (12901)
        #### FINAL TRAIN DATASET ###
        dataset = DKZ_Dataset(phospho_embedded, train_ds.KinaseEmbeddings, labels, train_ds.UniqueKinaseEmbeddings, is_train=True)

    elif mode in ['val', 'test']:
        val_test_ds =  all_seq_dataset()
        val_test_ds.get_data(data_path, KE, is_labeled=is_labeled, MultiLabel=True)

        ### Model Input (Val) ### (80, 13, 100)
        phospho_embedded = val_test_ds.get_embedded_seqs(AminoAcidProperties=False, ProtVec=True)

        if normalize_embedding:
            phospho_embedded_reshape = phospho_embedded.reshape(phospho_embedded.shape[0], phospho_embedded.shape[1] * phospho_embedded.shape[2])
            phospho_embedded_reshape = embed_scaler.transform(phospho_embedded_reshape)
            phospho_embedded = phospho_embedded_reshape.reshape(phospho_embedded.shape[0], phospho_embedded.shape[1], phospho_embedded.shape[2])
        
        # (17,), (17,727), (17,), (17,), (17,)
        if candidate_path is not None:
            
            candidate_kinase, candidate_kinase_embedding, candidate_indices, candidate_ke_to_kinase, candidate_uniprotid = KE.read_kinases_from_path(candidate_path)
            
            if is_labeled:
                val_labels = FindTrueClassIndices(val_test_ds.KinaseEmbeddings, candidate_kinase_embedding, True) # (80,), icindekiler 1 elemanlik list
                #### FINAL VAL DATASET ###
                dataset = DKZ_Dataset(phospho_embedded, val_test_ds.KinaseEmbeddings, val_labels, val_test_ds.UniqueKinaseEmbeddings, is_train=False)


    phosphosite_seq_size = all_seq_dataset.Get_SeqSize(AminoAcidProperties=False, ProtVec=True)

    if mode == 'train':
        return dataset, phosphosite_seq_size, embed_scaler
    elif mode in ['val', 'test']:
        return dataset, phosphosite_seq_size, KE, candidate_kinase_embedding, candidate_ke_to_kinase, val_test_ds.KinaseUniProtIDs, candidate_uniprotid