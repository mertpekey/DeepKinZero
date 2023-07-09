import argparse
import warnings
warnings.filterwarnings("ignore")
import os

from train import train_model
from test import test_model

def get_abs_path(file_name):
    return os.path.abspath(file_name)

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Training Parser')
    
    # Bunu positional argument yapabiliriz (Zorunlu yani) ve sadece train, test ve predict secilebilsin
    parser.add_argument('--MODE', type=str, default='train')

    parser.add_argument('--DATA_PATH', type=str, default=get_abs_path('Dataset/'))
    parser.add_argument('--TRAIN_DATA', type=str, default=get_abs_path('Dataset/Train_Phosphosite.txt'))
    parser.add_argument('--VAL_DATA', type=str, default=get_abs_path('Dataset/Val_Phosphosite_MultiLabel.txt'))
    parser.add_argument('--VAL_KINASE_CANDIDATES', type=str, default=get_abs_path('Dataset/Val_Candidate_Kinases.txt'))
    parser.add_argument('--TEST_DATA', type=str, default=get_abs_path('Dataset/Test_Phosphosite_MultiLabel.txt'))
    parser.add_argument('--TEST_KINASE_CANDIDATES', type=str, default=get_abs_path('Dataset/AllCandidates.txt'))
    parser.add_argument('--TEST_IS_LABELED', type=bool, default=True)
    parser.add_argument('--KINASE_PATH', type=str, default=get_abs_path('Dataset/KinaseFeatures.txt'))
    parser.add_argument('--KINASE_EMBEDDING_PATH', type=str, default=get_abs_path('Dataset/KinaseFeatures.txt'))

    parser.add_argument('--DEVICE', type=str, default='cpu')
    parser.add_argument('--NUM_OF_MODELS', type=int, default=1)
    parser.add_argument('--NUM_EPOCHS', type=int, default=50)
    parser.add_argument('--MODEL_TYPE', type=str, default='ESM') # [BiLSTM, ProtBERT, Transformer_LSTM, ESM]
    parser.add_argument('--BATCH_SIZE', type=int, default=64)
    parser.add_argument('--LEARNING_RATE', type=float, default=0.001)
    parser.add_argument('--SEQ_SIZE', type=int, default=7)
    parser.add_argument('--IS_HUGGINGFACE', type=bool, default=False)
    parser.add_argument('--HF_ONLY_ID', type=bool, default=True)
    
    parser.add_argument('--SAVE_MODEL', type=bool, default=False)
    parser.add_argument('--SAVE_FILEPATH', type=str, default='50Epochs_DKZ')

    parser.add_argument('--ESM_MODEL_NAME', type=str, default='esm2_t6_8M_UR50D')
    parser.add_argument('--USE_ESM_KINASE', type=bool, default=False)
    parser.add_argument('--USE_ESM_PHOSPHOSITE', type=bool, default=True)

    args = parser.parse_args()


    if args.MODE == 'train':
        train_model(args)
    elif args.MODE == 'test':
        test_model(args)
    elif args.MODE == 'predict':
        pass
    else:
        print('Give valid MODE arguments. Valid MODEs are train, test or predict')
