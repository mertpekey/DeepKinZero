from create_dataset import create_datasets
from utils import ensemble, get_eval_predictions
from sklearn.metrics import classification_report
from sklearn.preprocessing import MultiLabelBinarizer
import torch
import numpy as np

def test_model(args):
    pass
    # Burayi duzenleyecegim


    # Results (Test)
    if args.TEST_DATA != '':

        # Load Models
        loaded_models = []

        # Load Embed Scaler
        embed_scaler = None
       
        test_dataset, _, KE, test_candidate_kinase_embeddings, test_candidate_ke_to_kinase, test_kinase_uniprotid, test_candidate_uniprotid = create_datasets(args.TEST_DATA, args.TEST_KINASE_CANDIDATES, mode='test',embed_scaler=embed_scaler)

        mlb_test = MultiLabelBinarizer()
        binlabels_true_test = mlb_test.fit_transform(test_kinase_uniprotid)

        for model in loaded_models:
            TestUniProtIDs, test_probabilities = eval_test(test_dataset.DE, test_candidate_kinase_embeddings, test_candidate_ke_to_kinase)
            TestUniProtIDs, test_probabilities = ensemble(TestUniProtIDs, test_probabilities, test_candidate_uniprotid)
            Test_Evaluation, binlabels_pred = get_eval_predictions(TestUniProtIDs, test_probabilities, test_kinase_uniprotid, test_dataset.TCI, mlb_test)
            print('Acccuracy_Test: {}  Loss_Test: {} Top3Accuracy: {} Top5Accuracy: {} Top10Accuracy: {}'.format(Test_Evaluation["Accuracy"], Test_Evaluation["Loss"], Test_Evaluation["Top3Acc"], Test_Evaluation["Top5Acc"], Test_Evaluation["Top10Acc"]))
            print(classification_report(binlabels_true_test, binlabels_pred, target_names=mlb_test.classes_))



def eval_test(model,
              test_dataloader,
              TestCandidatekinaseEmbeddings,
              TestCandidateKE_to_Kinase,
              TestKinaseUniProtIDs,
              mlb_test,
              args):
    
    model.eval()
        
    #X, CE, y = next(iter(val_dataloader))
    X, CE, y = test_dataloader.DE, test_dataloader.ClassEmbedding_with1, test_dataloader.TCI
    
    TestCandidateKinases_with1 = torch.from_numpy(np.c_[ TestCandidatekinaseEmbeddings, np.ones(len(TestCandidatekinaseEmbeddings))]).float()
    
    X.to(args.DEVICE)
    TestCandidateKinases_with1.to(args.DEVICE)

    allUniProtIDs = []
    allprobs = []

    with torch.no_grad():

        pred = model(X)

        # Equation 5 from paper
        logits = torch.matmul(pred, TestCandidateKinases_with1.T)
        outclassidx = torch.argmax(logits, dim=1) 
        classes = TestCandidatekinaseEmbeddings[outclassidx]
        probabilities = torch.nn.functional.softmax(logits, dim=1)

    # get UniProtIDs for predicted classes and return them
    UniProtIDs =[]
    for c in classes:
        UniProtIDs.append(TestCandidateKE_to_Kinase[tuple(c)])
    UniProtIDs = np.array(UniProtIDs)

    Test_Evaluation, binlabels_pred = get_eval_predictions(UniProtIDs, probabilities, TestKinaseUniProtIDs, y, mlb_test)

    return Test_Evaluation, UniProtIDs, probabilities, binlabels_pred