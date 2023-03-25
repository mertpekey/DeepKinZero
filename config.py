import os

TRAIN_DATA = 'Dataset/Train_Phosphosite.txt'
VAL_DATA = 'Dataset/Val_Phosphosite_MultiLabel.txt'
VAL_KINASE_CANDIDATES = 'Dataset/Val_Candidate_Kinases.txt'
TEST_DATA = 'Dataset/PhosPhoELM/PhoELMdata.txt'
TEST_KINASE_CANDIDATES = 'Dataset/AllCandidates.txt'

DATA_PATH = '/Dataset'
KINASE_PATH = os.path.join(DATA_PATH,"AllKinases.txt")
KINASE_EMBEDDING_PATH = os.path.join(DATA_PATH,"KinaseFeatures.txt")
AMINOACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '_']
SEQ_SIZE = 7 # Sequence size of phosphosite s + 1 + s
BATCH_SIZE = 64
LEARNING_RATE = 0.001
CLIP_GRADIENTS = 9.0
RNN_UNIT_TYPE = 'LNlstm'
NUM_LSTM_LAYERS = 2
NUM_HIDDEN_UNITS = 512
DROPOUT_VAL = 0.5
EMBEDDING_DIM: 500
ATTENTION_SIZE: 20
REGS = 0.001
NUM_OF_MODELS = 10