# prepare sequence embeddings of Rostlab/prot_t5_xl_uniref50
from transformers import T5Tokenizer, T5EncoderModel, BertModel, BertTokenizer
import torch
import csv
import time
import numpy as np
import pickle


def get_sequence(file):
    Sequences = []
    with open(file) as csvfile:
        Sub_DS = csv.reader(csvfile, delimiter='\t')
        for row in Sub_DS:
            Sequences.append(row[2].upper())
    return Sequences

def get_embedding(Sequences, model, tokenizer):
    embs = {}

    for seqs in Sequences:
        for seq in seqs:
            if seq not in embs:
                encoded_input = tokenizer.batch_encode_plus(seq, return_tensors='pt')
                with torch.no_grad():
                    outputs = model(torch.tensor(encoded_input["input_ids"]))

                # extract embeddings for the  sequence in the batch while removing padded & special tokens ([0,:15]) 
                embs[seq] = outputs.last_hidden_state.tolist()


    return embs

start_time = time.time()
#tokenizer = T5Tokenizer.from_pretrained("Rostlab/prot_t5_xl_uniref50", do_lower_case=False )
#model = T5EncoderModel.from_pretrained("Rostlab/prot_t5_xl_uniref50")
tokenizer = BertTokenizer.from_pretrained("Rostlab/prot_bert", do_lower_case=False )
model = BertModel.from_pretrained("Rostlab/prot_bert")

SequencesTr = get_sequence("Dataset/Train_Phosphosite.txt")
SequencesVal = get_sequence("Dataset/Val_Phosphosite_MultiLabel.txt")
SequencesTest = get_sequence("Dataset/Test_Phosphosite_MultiLabel.txt")
Sequences = [SequencesTr, SequencesVal, SequencesTest]
Embeddings = get_embedding(Sequences, model, tokenizer)

with open('pretrained_files/embeddings/ProtT5xlSeqEmb.pickle', 'wb') as handle:
    pickle.dump(Embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

#with open('pretrained_files/embeddings/ProtT5xlSeqEmb.pickle', 'wb') as handle:
#    pickle.dump(Embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

"""
with open('pretrained_files/embeddings/ProtT5xlSeqEmb.pickle', 'rb') as handle:
    b = pickle.load(handle)

print(len(b))
"""

"""
writef = open("pretrained_files/embeddings/ProtT5xlSeqEmb.txt", "a")

for seq, emb in Embeddings.items():
    writef.write(seq + " " + emb)
    writef.write("\n")
"""
print("--- %s seconds ---" % (time.time() - start_time))

