import numpy as np

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


## These are not going to implemented as function but written to remember to use later
def LinearActivation(_x):
    pass
def prelu(_x, scope=None):
    pass