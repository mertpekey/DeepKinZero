import numpy as np
import os
import csv

import config as config
from data.amino_acids import AminoAcids

import torch
from transformers import T5Tokenizer, T5EncoderModel, BertModel, BertTokenizer


class all_seq_dataset:
    """
    This class holds the data of substrates and their sequences, these can be pairs of labeled instances or unlabeled.
    It also generates different data embeddings.
    """

    def __init__(self):
        """
        The construct only reads trigram vectors to initialize the dictionaries
        """
        
        self.AminoAcids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '_']

        self.Sequences= [] # The sequences of sites
        self.Sub_IDs= [] # The Uniprot IDs of Substrates
        self.Residues = []
        self.Kinases= [] # The list of kinases it can be empty if this is a test set
        self.SeqSize = config.SEQ_SIZE # The sequence size of the phosphosite is SeqSize * 2 + 1 this paramether indicates how many of amino acids we will pick in each side of the site

        # KINASE ATTRIBUTES
        self.Kin_One_Hot_Encoded= [] #The one hot encoded kinases will be used for training RNN
        self.Kinase_ToOneHot = {} #A dictionary for converting kinase class to one hot vector
        self.OneHot_ToKinase = {} #A dictionary for converting one hot one hot vector to kinase class
        self.UniqueKinases = [] #Unique kinases in this dataset the type is kinase classes
        self.UniqueKinaseEmbeddings = [] #Unique kinase embeddings
        self.UniqueKinaseCounts = [] #The count of each unique kinase in this dataset
        self.Num_of_UniqKin = 0 #Number of unique kinases in this dataset
        self.KinaseEmbeddings = [] #The Kinase embeddings of the kinases in this class
    
        # PHOSPHOSITE ATTRIBUTES
        self.TrigramToVec = {} #A dictionary holding the ProtVec vector for each possible trigram of amino acids
    
        self.protVecVectors = [] #A list of protvec vectors for each sequence in the dataset
        self.seqBinaryVectors = []
        self.propertyVectors = []
        self.transformerVectors = []

        self.tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False)
        
        self.read_trigram_vectors()

        
    def get_data(self, datapath, AllKinases, is_labeled=False, MultiLabel=False, is_huggingface = False):
        """
        This methods reads the given data and fills the paramethers
        
        Args:
            datapath (string): the path of the dataset file, the file should be a csv file and its columns should be substrate_ID, position, sequence, Kinase_UniprotID(optional)
            is_labeled (bool): boolean value shows if the dataset have labels or not (the labels should be on last column)
        """

        # Creating Attributes
        self.MultiLabel = MultiLabel
        self.protVecVectors = []
        self.Kinases = []
        self.KinaseUniProtIDs = []

        # Reading Data
        with open(datapath) as csvfile:
            Sub_DS = csv.reader(csvfile, delimiter='\t')
            
            for row in Sub_DS:
                if is_labeled and not MultiLabel:
                    # Check if the kinase is in All_kinases file, if it is not, skip the line
                    if row[3] not in AllKinases.UniProtID_to_Kinase:
                        print(row[3], " Not Found in Allkinases dataset skipping row:", row)
                        continue

                self.Sub_IDs.append(row[0])
                self.Residues.append(row[1])

                if is_huggingface:
                    Sequence = row[2].upper()
                else:
                    Sequence = list(row[2].upper())
                self.Sequences.append(Sequence)

                if is_labeled:
                    if not MultiLabel:
                        kinase = AllKinases.UniProtID_to_Kinase[row[3]]
                        self.Kinases.append(kinase)
                        self.KinaseUniProtIDs.append(row[3])
                        if kinase not in self.UniqueKinases:
                            self.UniqueKinases.append(kinase)
                    else:
                        curkinases = []
                        UniProtIDs = row[3].split(',')
                        for id in UniProtIDs:
                            kinase = AllKinases.UniProtID_to_Kinase[id]
                            curkinases.append(kinase)
                            if kinase not in self.UniqueKinases:
                                self.UniqueKinases.append(kinase)
                        self.Kinases.append(curkinases)
                        self.KinaseUniProtIDs.append(UniProtIDs)

                
                if is_huggingface == False:
                    vec_mat = self.get_protvec_vectors(Sequence)
                    self.protVecVectors.append(self.pad_vectors(Sequence, vec_mat))

            if is_labeled:
                self.create_onehot_kinase_embeddings()

            if is_huggingface == False:
                self.protVecVectors = np.array(self.protVecVectors) # (12901,13,100)
                self.set_onehot_encodedseq(self.Sequences)
                AA = AminoAcids()
                self.propertyVectors, _ = AA.get_onehot_allalphabet(self)
            else:
                # Creating tokens for pretrained transformer model
                if config.HF_ONLY_ID:
                    for seq in self.Sequences:
                        seq = ' '.join(list(seq)) # ProtBERT
                        encoded_input = self.tokenizer(seq, return_tensors='pt') # Bu tum sequence'i aliyor
                        #encoded_input = self.tokenizer.batch_encode_plus(seq, return_tensors='pt') # Bu tek tek aminoacidleri
                        self.transformerVectors.append(encoded_input['input_ids'])
                    self.transformerVectors = torch.stack(self.transformerVectors)
                else:
                    temp_seq_list = [' '.join(list(seq)) for seq in self.Sequences]
                    self.transformerVectors = self.tokenizer(temp_seq_list, return_tensors='pt')
                print()


    

    ### PROTVEC METHODS ###

    def read_trigram_vectors(self):
        """
        Read the file which contains the protvec vectors for all the possible trigrams and store it in a dictionary
        """
        with open(os.path.join(config.DATA_PATH, 'Allcomb2Vec.txt'), 'r') as f:
            for line in f:
                line = line.rstrip()
                splits = line.split('\t')
                Trigram = splits[0]
                splitted = splits[1].split(" , ")
                splitted = list(map(float, splitted))
                self.TrigramToVec[Trigram] = splitted
    

    def get_protvec_vectors(self, Sequence):
        """
        Given a sequence return the protvec vectors for each trigram
        
        Args:
            Sequence (str):The sequence of amino acids to generate trigrams for
        
        Return:
            vecmat (list): list of protvec vectors for each trigram in the sequence, the size of each protvec vector is 100
        """
        vec_mat = []
        trigrams = self.get_trigrams(Sequence)
        for grams in trigrams:
                if "_" not in grams:
                    vec_mat.append(self.TrigramToVec[''.join(grams)])
        return vec_mat
    
    def get_trigrams(self, Sequence):
        """
        Return all trigrams of a sequence
        
        Args:
            Sequence (str): The sequence to generate the trigrams for
        
        Return:
            out (list): list of all trigrams
        """
        out = []
        for i in range(len(Sequence)-2):
            out.append(Sequence[i:(i+3)])
        return out
    
    def pad_vectors(self, Sequence, vec):
        """
        Method to pad ProtVec Vectors based on '_' in the sequence 
        so if there is any '_' in any of the trigrams it will be restored with a array of zeros of length 100
        
        Args:
            Sequence (str): The input sequence
            vec (list): The input ProtVec vector
            
        Return:
            outvec (str): The padded ProtVec vector
        """
        outvec = []
        j = 0
        for i in range(len(Sequence) - 2):
            if '_' in Sequence[i:i+3]:
                outvec.append(np.zeros(100).tolist())
            else:
                outvec.append(vec[j])
                j += 1
        return outvec
    
    ### ONE HOT ENCODING ###

    def set_onehot_encodedseq(self, Sequences):
        """
        create one-hot binary encoded sequence for each sequence in the Sequences array and set it as binarryseqarray
        
        Args:
            Sequences (list): list of sequences of amino acids
        """
        self.integer_encoded_seqs = []
        for seq in Sequences:
            intseq = []
            for c in seq:
                intseq.append(self.AminoAcids.index(c))
            self.integer_encoded_seqs.append(intseq)
        self.integer_encoded_seqs = np.array(self.integer_encoded_seqs) # (12901, 15)
        self.seqBinaryVectors = self.get_onehot_encodedseq(self.AminoAcids, Sequences)
    
    def get_onehot_encodedseq(self, chars, Sequences):
        """
        Create one-hot binary sequence for each sequence in sequences and returns it
        
        Args:
            Sequences (list): list of sequences of amino acids
        """
        alphabet = sorted(set(chars))
        onehottedseq = []

        for seq in Sequences:
            onehottedseq.append(self.string_vectorizer(seq, alphabet))

        onehotseqarray = []
        for i in range(0,len(onehottedseq)):
            onehotseqarray.append(np.concatenate(onehottedseq[i],0))
        onehotseqarray = np.array(onehotseqarray)

        return onehotseqarray
    
    def string_vectorizer(self,strng, alphabet):
        vector = [[0 if char != letter else 1 for char in alphabet] 
                      for letter in strng]
        return vector
    

    def create_onehot_kinase_embeddings(self):
        """
        Create One hot class embedding for each kinase and store them in the parameters in the class
        """
        #self.UniqueKinases, self.UniqueKinaseCounts = np.unique(self.Kinases,return_counts=True)
        self.Num_of_UniqKin = len(self.UniqueKinases)
        

        for i, UniqKin in enumerate(self.UniqueKinases):
            OneHot = np.zeros([self.Num_of_UniqKin])
            OneHot[i] = 1
            self.Kinase_ToOneHot[UniqKin] = OneHot
            self.UniqueKinaseEmbeddings.append(UniqKin.EmbeddedVector)

        for kin in self.Kinases:
            if not self.MultiLabel:
                self.Kin_One_Hot_Encoded.append(self.Kinase_ToOneHot[kin])
                self.KinaseEmbeddings.append(kin.EmbeddedVector)
            else:
                onehots = []
                kinase_embeddings = []
                for k in kin:
                    onehots.append(self.Kinase_ToOneHot[k])
                    kinase_embeddings.append(k.EmbeddedVector)
                self.Kin_One_Hot_Encoded.append(onehots)
                self.KinaseEmbeddings.append(kinase_embeddings)

        self.KinaseEmbeddings = np.array(self.KinaseEmbeddings) # (12901,727)
        self.Kin_One_Hot_Encoded = np.array(self.Kin_One_Hot_Encoded) # (12901, 214)
        self.UniqueKinaseEmbeddings = np.array(self.UniqueKinaseEmbeddings) # (214, 727)


    ### DATA USED FOR MODEL ###

    def get_input_embeddings(self, AminoAcidProperties, ProtVec, Transformer):
        """
        Get the embedded sequences as a list of vectors
        
        Args:
            AminoAcidProperties (binary): return sequences as represented by one-hot binary vectors for each amino acid
            ProtVec (binary): return sequences as represented by protvec vectors
        
        Return:
            a list of vectors generated for each sequence in the dataset, the length of vectors is different based on the selected method
            Protvec will generate vectors of length 100, AminoAcidProperties will generate vectors of length  and binary will generate vectors of length 315
        """
        if Transformer:
            return self.transformerVectors
        elif ProtVec:
            return self.protVecVectors
        elif AminoAcidProperties:
            return self.propertyVectors
        else:
            return self.seqBinaryVectors.reshape([-1, 15, 21])

    @staticmethod
    def Get_SeqSize(AminoAcidProperties, ProtVec, Transformer):
        """
        According to given flags AminoAcidProperties and ProtVec, calculates the size of output
        The output is a tuple of (Seq_size, EmbeddingSize)
        
        Args:
            AminoAcidProperties (binary): return sequences as represented by one-hot binary vectors for each amino acid
            ProtVec (binary): return sequences as represented by protvec vectors
        
        Return:
            a tuple of sequence (SequenceSize, EmbeddingSize)
        """
        if Transformer:
            return (17,1024)
        elif ProtVec:
            return (13, 100)
        elif AminoAcidProperties:
            return (15, 16)
        else:
            return (15, 21)