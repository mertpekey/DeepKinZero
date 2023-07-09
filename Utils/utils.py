import numpy as np
#from esm.pretrained import load_model_and_alphabet as esm_load_model_and_alphabet
import torch

def ensemble(UniProtIDs, probabilities, CandidatekinaseUniProtIDs):
    sum_probs = np.sum(np.array(probabilities), axis=0) / len(probabilities)
    outclassindx = np.argmax(sum_probs, axis=1)
    CandidatekinaseUniProtIDs = np.array(CandidatekinaseUniProtIDs)
    outUniprotIDs = CandidatekinaseUniProtIDs[outclassindx]
    return outUniprotIDs, sum_probs


def get_top_n(n, matrix):
    """Gets probability a number n and a matrix, 
    returns a new matrix with largest n numbers in each row of the original matrix."""
    return (-matrix).argsort()[:,0:n]


def intersection(lst1, lst2): 
    return list(set(lst1) & set(lst2)) 


def GetAccuracyMultiLabel(Y_Pred, Probabilities, Y_True, TrueClassIndices, eps=1e-15):
        """ Returns Accuracy when multi-label are provided for each instance. It will be counted true if predicted y is among the true labels
        Args:
            Y_Pred (int array): the predicted labels
            Probabilities (float [][] array): the probabilities predicted for each class for each instance
            Y_True (int[] array): the true labels, for each instance it should be a list
        """
        count_true = 0
        count_true_3 = 0
        count_true_5 = 0
        count_true_10 = 0
        logloss = 0.0
        top_10_classes = get_top_n(10, Probabilities)
        for i in range(len(Y_Pred)):
            for idx in TrueClassIndices[i]:
                p = np.clip(Probabilities[i][idx], eps, 1 - eps)
                logloss -= np.log(p)
            if Y_Pred[i] in Y_True[i]:
                count_true += 1
            if len(intersection(top_10_classes[i], TrueClassIndices[i])) > 0:
                count_true_10 += 1
            if len(intersection(top_10_classes[i][:5], TrueClassIndices[i])) > 0:
                count_true_5+=1
            if len(intersection(top_10_classes[i][:3], TrueClassIndices[i])) > 0:
                count_true_3+=1
        
        Evaluations = {"Accuracy":(float(count_true) / len(Y_Pred)), "Loss":(float(logloss) / len(Y_Pred)), 
                       "Top3Acc":(float(count_true_3) / len(Y_Pred)) ,"Top5Acc":(float(count_true_5) / len(Y_Pred)) ,
                       "Top10Acc":(float(count_true_10) / len(Y_Pred)), "Probabilities":Probabilities, 
                       "Ypred":Y_Pred, "Ytrue":Y_True}
        return Evaluations


def get_eval_predictions(UniProtIDs, probabilities, ValKinaseUniProtIDs, y, mlb_Val):
    
    predlabels = [[label] for label in UniProtIDs]
    binlabels_pred = mlb_Val.transform(predlabels)
    Val_Evaluation = GetAccuracyMultiLabel(UniProtIDs, probabilities, ValKinaseUniProtIDs, y)

    return Val_Evaluation, binlabels_pred

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


### --- ESM Utils ---

def load_esm_model(model_name):
    model, alphabet = torch.hub.load("facebookresearch/esm:main", model_name)
    #model, alphabet = esm_load_model_and_alphabet(model_name)
    return model, alphabet

def get_esm_embedding_dim(model_name):
    esm_info = {
        'esm2_t48_15B_UR50D':5120,
        'esm2_t36_3B_UR50D':2560,
        'esm2_t33_650M_UR50D':1280,
        'esm2_t30_150M_UR50D':640,
        'esm2_t12_35M_UR50D':480,
        'esm2_t6_8M_UR50D':320,
        'esm_if1_gvp4_t16_142M_UR50':512,
        'esm1v_t33_650M_UR90S_[1-5]':1280,
        'esm_msa1b_t12_100M_UR50S':768,
        'esm1b_t33_650M_UR50S':1280,
        'esm1_t34_670M_UR50S':1280,
        'esm1_t34_670M_UR50D':1280,
        'esm1_t34_670M_UR100':1280,
        'esm1_t12_85M_UR50S':768,
        'esm1_t6_43M_UR50S':768
    }
    if model_name.startswith('esm1v_t33_650M_UR90S_'):
        model_name = 'esm1v_t33_650M_UR90S_[1-5]'
    return esm_info[model_name]