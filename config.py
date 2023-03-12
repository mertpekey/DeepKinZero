import os

DATA_PATH = '/Dataset'
KINASE_PATH = os.path.join(DATA_PATH,"AllKinases.txt")
KINASE_EMBEDDING_PATH = os.path.join(DATA_PATH,"KinaseFeatures.txt")
AMINOACIDS = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y', '_']
SEQ_SIZE = 7 # Sequence size of phosphosite s + 1 + s