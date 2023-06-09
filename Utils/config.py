import os

def get_abs_path(file_name):
    return os.path.abspath(file_name)


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

TRANSFORMER_HIDDEN_UNITS = 1024
TRANSFORMER_EMBEDDING_TYPE = 'POOLER'
HF_ONLY_ID = True
USE_SECOND_TRANSFORMER = False