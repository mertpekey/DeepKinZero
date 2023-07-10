from data.dataset import DKZ_Dataset
from data.sequence_data import all_seq_dataset

from Utils.utils import FindTrueClassIndices
from data.kinase_embeddings import KinaseEmbedding

from sklearn import preprocessing
import numpy as np


def create_datasets(data_path, args, candidate_path = None, mode='train', is_labeled=True, normalize_embedding = True, embed_scaler=None):
    # Define Dataset
    KE = KinaseEmbedding(args=args, Family = True, Group = True, Pathway = False, Kin2Vec=True, Enzymes = True)

    if mode == 'train':
        # Create the training dataset
        train_ds = all_seq_dataset(args)
        train_ds.get_data(data_path, KE, is_labeled=is_labeled, args=args)
        
        ### Model Input (Train) ### Protvec: (12901,13,100), Huggingface: (12901, ?, ?), ESM (12901, 17)
        phospho_embedded = train_ds.get_input_embeddings(AminoAcidProperties=False,
                                                         ProtVec=True,
                                                         Transformer=args.IS_HUGGINGFACE,
                                                         ESM = args.USE_ESM_PHOSPHOSITE) # Get the sequence embeddings
        
        #Normalize Training data
        if normalize_embedding and args.IS_HUGGINGFACE == False and args.USE_ESM_PHOSPHOSITE == False:
            phospho_embedded_reshape = phospho_embedded.reshape(phospho_embedded.shape[0], phospho_embedded.shape[1] * phospho_embedded.shape[2])
            embed_scaler = preprocessing.StandardScaler().fit(phospho_embedded_reshape)
            phospho_embedded_reshape = embed_scaler.transform(phospho_embedded_reshape)
            phospho_embedded = phospho_embedded_reshape.reshape(phospho_embedded.shape[0], phospho_embedded.shape[1], phospho_embedded.shape[2])
        labels = FindTrueClassIndices(train_ds.KinaseEmbeddings, train_ds.UniqueKinaseEmbeddings) # KinaseEmb (12901,727), Unique (214,727), TCI (12901)
        #### FINAL TRAIN DATASET ###
        is_input_tensor = True if (args.IS_HUGGINGFACE or args.USE_ESM_PHOSPHOSITE) else False
        dataset = DKZ_Dataset(phospho_embedded, train_ds.KinaseEmbeddings, labels, train_ds.UniqueKinaseEmbeddings, args=args, is_train=True, is_input_tensor=is_input_tensor)

    elif mode in ['val', 'test']:
        val_test_ds =  all_seq_dataset(args)
        val_test_ds.get_data(data_path, KE, is_labeled=is_labeled, MultiLabel=True, args=args)

        ### Model Input (Val) ### (80, 13, 100)
        phospho_embedded = val_test_ds.get_input_embeddings(AminoAcidProperties=False,
                                                            ProtVec=True,
                                                            Transformer=args.IS_HUGGINGFACE,
                                                            ESM = args.USE_ESM_PHOSPHOSITE) # Get the sequence embeddings

        if normalize_embedding and args.IS_HUGGINGFACE == False and args.USE_ESM_PHOSPHOSITE == False:
            phospho_embedded_reshape = phospho_embedded.reshape(phospho_embedded.shape[0], phospho_embedded.shape[1] * phospho_embedded.shape[2])
            phospho_embedded_reshape = embed_scaler.transform(phospho_embedded_reshape)
            phospho_embedded = phospho_embedded_reshape.reshape(phospho_embedded.shape[0], phospho_embedded.shape[1], phospho_embedded.shape[2])
        
        # (17,), (17,727), (17,), (17,), (17,)
        if candidate_path is not None:
            
            candidate_kinase, candidate_kinase_embedding, candidate_indices, candidate_ke_to_kinase, candidate_uniprotid = KE.read_kinases_from_path(candidate_path)
            
            if is_labeled:
                if args.USE_ESM_KINASE:
                    max_size = max(val_test_ds.KinaseEmbeddings[0][0].shape[0], candidate_kinase_embedding.shape[1])
                    if max_size == val_test_ds.KinaseEmbeddings[0][0].shape[0]:
                        candidate_kinase_embedding = np.pad(candidate_kinase_embedding, [(0, 0), (0, max_size - candidate_kinase_embedding.shape[1])], mode='constant', constant_values=KE.esm_alphabet.padding_idx)
                    else:
                        #### BUNA BAK
                        val_test_ds.KinaseEmbeddings = np.pad(val_test_ds.KinaseEmbeddings[0][0], (0, max_size - val_test_ds.KinaseEmbeddings[0][0].shape[0]), mode='constant', constant_values=KE.esm_alphabet.padding_idx)
                        #val_test_ds.KinaseEmbeddings = np.pad(val_test_ds.KinaseEmbeddings, [(0, 0), (0, max_size - val_test_ds.KinaseEmbeddings.shape[1])], mode='constant', constant_values=KE.esm_alphabet.padding_idx)
                val_labels = FindTrueClassIndices(val_test_ds.KinaseEmbeddings, candidate_kinase_embedding, True) # (80,), icindekiler 1 elemanlik list
                #### FINAL VAL DATASET ###
                is_input_tensor = True if (args.IS_HUGGINGFACE or args.USE_ESM_PHOSPHOSITE) else False
                dataset = DKZ_Dataset(phospho_embedded, val_test_ds.KinaseEmbeddings, val_labels, val_test_ds.UniqueKinaseEmbeddings, args=args, is_train=False, is_input_tensor=is_input_tensor)

    # Input Data Size
    phosphosite_seq_size = all_seq_dataset.Get_SeqSize(AminoAcidProperties=False,
                                                       ProtVec=False,
                                                       Transformer=args.IS_HUGGINGFACE,
                                                       ESM=args.USE_ESM_PHOSPHOSITE,
                                                       ESM_model_name=args.ESM_MODEL_NAME)

    if mode == 'train':
        return dataset, phosphosite_seq_size, embed_scaler
    elif mode in ['val', 'test']:
        return dataset, phosphosite_seq_size, KE, candidate_kinase_embedding, candidate_ke_to_kinase, val_test_ds.KinaseUniProtIDs, candidate_uniprotid