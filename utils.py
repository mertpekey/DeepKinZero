import numpy as np


def Write_predictions(FilePath, probabilities, Sub_IDs, Sequences, Residues, Candidatekinases, Candidate_UniProtIDs, top_n, sub_ID_has_original_rows=False):
    with open(FilePath, 'w') as predictions:
        print('Row id in the Original File, Phosphosite Residue, Substrate protein, Phosphosite Sequence' + ',' + ','.join(['Predicted Kinase UniProt ID', 'Predicted Kinase Name', 'Predicted Probability'] * top_n), file=predictions)
        original_row = 0
        for Probs, subID, Seq, Res in zip(probabilities, Sub_IDs, Sequences, Residues):
            sorted_predictions = [[x,y] for y, x in sorted(zip(Probs,Candidatekinases), key=lambda pair: pair[0], reverse=True)]
            if sub_ID_has_original_rows:
                sub_ID_splitted = subID.split('_')
                subID = sub_ID_splitted[1]
                original_row = sub_ID_splitted[0]
            else:
                original_row += 1
            row = str(original_row) + ',' + Res + ',' + subID + ',' + ''.join(Seq) + ','
            for i in range(top_n):
                prob = sorted_predictions[i][1]
                kinase = sorted_predictions[i][0]
                row += kinase.UniprotID + ',' + kinase.Protein_Name + ',' + str(prob) + ','
            print(row, file=predictions)

def FindTrueClassIndices(ClassEmbeddings, CandidateKinases, Multilabel = False):
    """
    Find the indices of the true classes in the candidate class embeddings
    Args:
        ClassEmbeddings: The class embeddings to find their indices
        CandidateKinases: The list of candidate kinases
    
    Returns:
        a list of index per entry in ClassEmbeddings showing its index in CandidateKinases
    """
    
    TrueClassIDX = []
    for CEW1 in ClassEmbeddings:
        if not Multilabel:
            idx = np.where(np.all(np.equal(CEW1, CandidateKinases), axis=1))
            TrueClassIDX.append(idx[0][0])
        else:
            indices = []
            for CE in CEW1:
                idx = np.where(np.all(np.equal(CE, CandidateKinases), axis=1))
                indices.append(idx[0][0])
            TrueClassIDX.append(indices)
    TrueClassIDX = np.array(TrueClassIDX)
    return TrueClassIDX

def getStrParam(Params):
    out = "UT={}_NL={}_NH={}_DO={}_LR={}_At={}_AtS={}_BN1={}_BN2={}_DO1={}_DO2={}_Regs={}_LRD={}".format(
            Params["rnn_unit_type"], Params["num_layers"], Params["num_hidden_units"],
            Params["dropoutval"], Params["learningrate"], Params["useAtt"],
            Params["ATTENTION_SIZE"], Params["UseBatchNormalization1"], Params["UseBatchNormalization2"],
            Params["Dropout1"], Params["Dropout2"], Params["regs"], Params["LRDecay"])
    

def createFolderName(OtherAlphabets, Gene2Vec, Family, Group, Pathways, Kin2Vec, Enzymes, Params, MainModel, EmbeddingOrParams=False):
    """ Creates two strings representing the parameters given for storing logs and results one of the strings is the name of the folder for ZSL results and the other one for saving data embedding models logs and models
    Args:
        OtherAlphabets (bool): Use other properties of Amino Acids
        Gene2Vec (bool): Use Gene2Vec for converting protein sequences into vector of continuos numbers
        RNNfeatures (bool): Use RNN as DataEmbedding tool
        BiRNNfeatures (bool): Use BiRNN as DataEmbedding tool
        BiRNNfocusfeatures (bool): Use attention mechanism in BiRNN
        Family (bool): Use Kinase family data in Class Embeddings
        Group (bool): Use Kinase group (superfamily) data in Class Embeddings
        STY (bool): Use Kinase type data (S T or Y) in Class Embeddings
        Pathways (bool): Use KEGG pathways data in Class Embeddings
        ConvLayers (int array): Number of Convolutional Layers to use in BiRNN
        MainModel (str): What is the main model for classification (it can be SZSL, SVM, LogisticRegression, RNN, BiRNN)
        
    Returns:
        Name (str): The name of the folder for ZSL results
        DEFolder (str): The name of the folder for Data Embedding model logs and checkpoints
    """
    DEFolder = ""
    if EmbeddingOrParams:
        Name = MainModel
        if Params["Bidirectional"]:
            Name += "BiRNN-"
            DEFolder = DEFolder + "BiRNN-"
        if Params["useAtt"]:
            Name += "Att-"
            DEFolder = DEFolder + "Att-"
        if OtherAlphabets:
            Name += "OA-"
            DEFolder = DEFolder + "OA-"
        if Gene2Vec:
            Name += "G2V-"
            DEFolder = DEFolder + "G2V-"
        if Family:
            Name += "Fam-"
        if Group:
            Name += "Group-"
        if Pathways:
            Name += "Path-"
        if Kin2Vec:
            Name += "K2V-"
        if Enzymes:
            Name += "Enz-"
        if len(Params["num_of_Convs"])>0:
            Name = Name + str(Params["ConvLayers"]) + "Conv"
        return Name, DEFolder
    else:
        return getStrParam(Params), DEFolder
