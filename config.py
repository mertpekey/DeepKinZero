import os

def get_abs_path(file_name):
    return os.path.abspath(file_name)

TRAIN_DATA = get_abs_path('Dataset/Train_Phosphosite.txt')
VAL_DATA = get_abs_path('Dataset/Val_Phosphosite_MultiLabel.txt')
VAL_KINASE_CANDIDATES = get_abs_path('Dataset/Val_Candidate_Kinases.txt')
TEST_DATA = get_abs_path('Dataset/Test_Phosphosite_MultiLabel.txt')
TEST_KINASE_CANDIDATES = get_abs_path('Dataset/AllCandidates.txt')
TEST_IS_LABELED = True

DEVICE = 'cpu' #cuda
DATA_PATH = get_abs_path('Dataset/')
KINASE_PATH = get_abs_path('Dataset/KinaseFeatures.txt')
KINASE_EMBEDDING_PATH = get_abs_path('Dataset/KinaseFeatures.txt')
AMINOACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '_']
SEQ_SIZE = 7 # Sequence size of phosphosite s + 1 + s
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CLIP_GRADIENTS = 9.0
RNN_UNIT_TYPE = 'LNlstm'
NUM_LSTM_LAYERS = 2
NUM_HIDDEN_UNITS = 512
DROPOUT_VAL = 0.5
EMBEDDING_DIM= 500
ATTENTION_SIZE= 20
REGS = 0.001
NUM_OF_MODELS = 1