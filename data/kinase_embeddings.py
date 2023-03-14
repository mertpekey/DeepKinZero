'''
This code is taken from https://github.com/tastanlab/DeepKinZero/blob/master/datautils.py
'''

# Numerical Operations
import numpy as np

# Preprocessing
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder

# Utils
import csv

# Files
import config as config



class Kinase:
    """
    Class for holding kinase information
    """
    def __init__(self, _Protein_Name,_EntrezID, _UniprotID, _EmbeddedVector = None):
        
        self.Protein_Name= _Protein_Name
        self.EntrezID= _EntrezID
        self.UniprotID= _UniprotID
        if _EmbeddedVector != None:
            self.EmbeddedVector = _EmbeddedVector
            
    def __lt__(self, otherKinase):
        """
        Making the kinase sortable by its name
        
        Args:
            otherKinase (kinase): Other kinase to compare to
        Return:
            True if the name of this kinase is smaller than the otherKinase
        """
        return (self.Protein_Name < otherKinase.Protein_Name)
    
    def __eq__(self, otherKinase):
        """
        Making the kinases comparable
        
        Args:
            otherKinase (kinase): Other kinase to compare to
        Return:
            True if kinases are equal and false otherwise
        """
        # it is enough to compare the uniprotIDs since it is unique for every protein
        return (self.UniprotID == otherKinase.UniprotID)
        #return (self.Protein_Name == otherKinase.Protein_Name)
        
    def __hash__(self):
        """
        Making the kinases hashable
        
        Return:
            hash of UniprotID
        """
        return hash(self.UniprotID)



class KinaseEmbedding:
    """
    This class holds the list of all kinases and create the kinase embeddings for each one
    """
    
    allkinases = [] # the list of all kinases
    UniProtID_to_Kinase = {} # a dictionary of uniprotID to kinase
    AllKinaseEmbeddings = [] # A list of all kinases embedded vectors
    KE_to_Kinase = {} # a dictionary of kinase Embedding vector to kinase class
    Embedding_size = 0 # The size of Kinase embeddings

    def __init__(self, Family = True, Group = True, Pathway = True, Kin2Vec = True, InterProDomains = True, Enzymes = True, SubCellLoc = True, GO_C_vec = True, GO_F_vec = True, GO_P_vec = True):
        """
        Get the paramethers for class embedding and initilize some variables to use later then run readKinaseEmbedding and ReadKinases methods to read all the kinases and create the embeddings for them
        
        Args:
            Family (bool): Use Kinase family data in Class Embeddings
            Group (bool): Use Kinase group (superfamily) data in Class Embeddings
            Pathway (bool): Use KEGG pathways data in Class Embeddings
            Kin2Vec (bool): Use Vectors generated from ProtVec for Kinase Sequences in Class Embeddings
            InterProDomains (bool): Use InterPro domains in Class Embedding
            Enzymes (bool): Use Enzymes hierarchy in Class Embedding
            SubCellLoc (bool): Use Sub cellular localisation in Class Embedding
            GO_C_vec: Use GO cellular component analysis in Class Embedding
            GO_F_vec: Use GO functional in Class Embedding
            GO_P_vec: Use GO pathways in Class Embedding
        """
        self.Family= Family
        self.Group=Group
        self.Pathway= Pathway
        self.Kin2Vec = Kin2Vec
        self.InterProDomains= InterProDomains
        self.Enzymes= Enzymes
        self.GO_C_vec= GO_C_vec
        self.GO_F_vec= GO_F_vec
        self.GO_P_vec= GO_P_vec
        self.read_kinase_embedding()
        self.read_kinases()
        self.Embedding_size = len(self.AllKinaseEmbeddings[0])
        
    def read_kinases(self):
        """
        This method reads all kinases from the AllKinasePath and creates the necessary dictionaries and arrays
        """
        with open(config.KINASE_PATH, 'r') as AllKinasefilecsv:
            AllKinasefile= csv.reader(AllKinasefilecsv, delimiter='\t')
            for row in AllKinasefile:
                newkinase = Kinase(row[0], row[1], row[2])
                self.allkinases.append(newkinase)
                self.UniProtID_to_Kinase[row[2]] = newkinase
            self.AllKinaseEmbeddings, _ = self.create_class_embedding(self.allkinases)

    def make_onehot_encoded(self,classes):
        """
        Gets an array and returns one hot encoded version of it
        
        Args:
            classes (string array): The input strings that will be one hot encoded
        """
        # Convert the array to numpy array
        values = np.array(classes)
        # Define a encoder
        label_encoder = LabelEncoder()
        # Convert the labels to integers
        integer_encoded = label_encoder.fit_transform(values)
        # binary encode
        onehot_encoder = OneHotEncoder(sparse=False)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
        
        return onehot_encoded, label_encoder

    def read_kinase_embedding(self):
        """
        Read the kinase embeddings file to get the features of kinases
        """
        with open(config.KINASE_EMBEDDING_PATH) as csvfile:
            KinEmb = csv.reader(csvfile, delimiter='\t')
            
            Families = []
            Groups = []
            KinUniProtIDs = []
            EnzymesVecs = []
            Domains = []
            Kin2Vecs = []
            Pathways = []

            # Read the kinase embedding file
            for row in KinEmb:
                KinUniProtIDs.append(row[0])
                Families.append(row[1])
                Groups.append(row[2])
                EnzymesVecs.append(list(map(int,list(row[3]))))
                Domains.append(row[4])
                Kin2Vecs.append(list(map(float, row[5].split(", "))))
                Pathways.append(list(map(int,map(float, row[6].split(", ")))))

            # Create dictionary for each of the embeddings
            self.Kinonehot_encoded, self.KinaseEncoder = self.make_onehot_encoded(KinUniProtIDs) # Convert UniProtIDs to one-hot encoded
            self.Familyonehot_encoded, self.familyEncoder = self.make_onehot_encoded(Families) # Convert families to one-hot encoded vectors
            self.Grouponehot_encoded, self.groupEncoder = self.make_onehot_encoded(Groups) # Convert groups to one-hot encoded vectors
            self.UniProtID_to_OneHotVec = dict(zip(KinUniProtIDs,self.Kinonehot_encoded))
            self.UniProtID_to_FamilyVec= dict(zip(KinUniProtIDs,self.Familyonehot_encoded))
            self.UniProtID_to_GroupVec= dict(zip(KinUniProtIDs,self.Grouponehot_encoded))
            self.UniProtID_to_Kin2Vec= dict(zip(KinUniProtIDs, Kin2Vecs))
            self.UniProtID_to_EnzymesVec = dict(zip(KinUniProtIDs, EnzymesVecs))
            self.UniProtID_to_Pathway= dict(zip(KinUniProtIDs, Pathways))
            
    def get_embedding(self, UniprotID):
        """
        Get embedding vector for a single kinase
        
        Args:
            UniprotID (int): UniProt ID of the kinase to generate its embedding
        
        Return:
            The embedded vector of the input kinase
        """
        ClassEmbedding = self.UniProtID_to_OneHotVec[UniprotID]
        if self.Group:
            ClassEmbedding = np.append(ClassEmbedding,self.UniProtID_to_GroupVec[UniprotID])
        if self.Family:
            ClassEmbedding = np.append(ClassEmbedding,self.UniProtID_to_FamilyVec[UniprotID])
        if self.Pathway:
            ClassEmbedding = np.append(ClassEmbedding,self.UniProtID_to_Pathway[UniprotID])
        if self.Kin2Vec:
            ClassEmbedding = np.append(ClassEmbedding,self.UniProtID_to_Kin2Vec[UniprotID])
        if self.Enzymes:
            ClassEmbedding = np.append(ClassEmbedding,self.UniProtID_to_EnzymesVec[UniprotID])
        return ClassEmbedding
    
    def create_class_embedding(self, kinases):
        """
        Get class embedding for each kinase in the given list and store them in kinase embedded vector and also return a list containing these class embeddings
        
        Args:
            kinases (list of kinase): The list of kinases to produce class embedding for
        
        Return:
            KinaseEmbeddings: list which is the sorted class embeddings for each of the input kinases
            UniqueClassEmbedding: list of unique class embeddings
        """
        KinaseEmbeddings = []
        for kin in kinases:
            kin.EmbeddedVector = self.get_embedding(kin.UniprotID)
            KinaseEmbeddings.append(kin.EmbeddedVector)
            self.KE_to_Kinase[tuple(kin.EmbeddedVector)] = kin
            
        UniqueClassEmbedding = np.vstack({tuple(row) for row in KinaseEmbeddings})
        return KinaseEmbeddings, UniqueClassEmbedding
    
    def read_kinases_from_path(self, path):
        """
        Read kinases from a given path
        
        Args:
            path(str): path of the file to read the kinases from. The file should contain a single column which contains the uniprotIDs of kinases
        
        Returns:
            kinases(list of kinase): a list of kinases in the file
            kinaseEmbeddings(list of list): a list of kinase embeddings
            indices(list of ints): indices of kinases in the allkinase file
        """
        indices = []
        kinases = []
        kinaseEmbeddings = []
        KE_to_Kinase = {}
        UniProtIDs = []
        with open(path, 'r') as kinasefile:
            for UniProtID in kinasefile:
                UniProtID = UniProtID.strip()
                UniProtIDs.append(UniProtID)
                kinase = self.UniProtID_to_Kinase[UniProtID]
                kinases.append(kinase)
                kinaseEmbeddings.append(kinase.EmbeddedVector)
                indices.append(self.allkinases.index(kinase))
                KE_to_Kinase[tuple(kinase.EmbeddedVector)] = UniProtID
        return np.array(kinases), np.array(kinaseEmbeddings), np.array(indices), KE_to_Kinase, UniProtIDs
    
    def get_UniProtIDs_from_KE(self, KinaseEmbeddings):
        """
        Get uniprotIDs of given kinase embeddings
        
        Args:
            KinaseEmbeddings(list of list): the list of kinase embeddings to return their uniprotIDs
        """
        UniProtIDs =[]
        for KE in KinaseEmbeddings:
            UniProtIDs.append(self.KE_to_Kinase[tuple(KE)].UniprotID)
        return np.array(UniProtIDs)