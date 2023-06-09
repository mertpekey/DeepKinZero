import torch
import torch.nn as nn
import Utils.config as config
from Utils.utils import load_esm_model, get_esm_embedding_dim

# rnnlib package for Bidirectional Layer Norm LSTM
# https://github.com/daehwannam/pytorch-rnn-library
from rnnlib.seq import LayerNormLSTM

# Transformer
from transformers import BertModel


###################### BiLSTM

class Bi_LSTM(nn.Module):
    def __init__(self, vocabnum, seq_lens, ClassEmbeddingsize):
        super(Bi_LSTM, self).__init__()

        self.vocabnum = vocabnum
        self.seq_lens = seq_lens
        self.ClassEmbeddingsize = ClassEmbeddingsize
        self.num_directions = 2 # Bidirectional

        # (1025, 728) # In paper, author mentions W is uniformly distributed
        self.W = torch.nn.Parameter(torch.rand(config.NUM_HIDDEN_UNITS * 2 + 1, self.ClassEmbeddingsize + 1) * 0.05)
        # Attention
        self.attention = Attention(config.ATTENTION_SIZE, config.NUM_HIDDEN_UNITS * 2)

        self.batchnorm1 = nn.BatchNorm1d(self.vocabnum)
        self.dropout_layer = nn.Dropout(p=0.5)##nn.Dropout1d(p=0.5)
        
        self.bi_lstm = LayerNormLSTM(self.vocabnum, config.NUM_HIDDEN_UNITS, config.NUM_LSTM_LAYERS, dropout=0, r_dropout=0,
                             bidirectional=True, layer_norm_enabled=True)

        self.batchnorm2 = nn.BatchNorm1d(config.NUM_HIDDEN_UNITS * 2)

    def forward(self,batch_embedded):
        # batch_embedded (n, l, c) -> (n, c, l)
        batch_embedded = batch_embedded.permute(0,2,1)
        x = self.batchnorm1(batch_embedded)
        x = self.dropout_layer(x)
        x = x.permute(2,0,1) # (n, c, l) -> (l, n, c)
        # 13,64,100
        x, _ = self.bi_lstm(x, None)
        # 13,64,1024
        x = x.permute(1,2,0) # (l, n, c) -> (n, c, l)
        x = self.batchnorm2(x)
        x = x.permute(0,2,1) # (n, c, l) -> (n, l, c)
        x, _ = self.attention(x)
        x = self.dropout_layer(x.unsqueeze(2)).squeeze(2) # (n, c, l)

        # Equation 3 from paper
        embedding = torch.nn.functional.pad(x, (0, 1), value=1)
        Matmul = torch.matmul(embedding, self.W)

        return Matmul


class Attention(nn.Module):
    def __init__(self, attention_size, hidden_size):
        super(Attention, self).__init__()
        self.attention_size = attention_size
        self.hidden_size = hidden_size

        # Trainable parameters
        self.w_omega = nn.Parameter(torch.randn(self.hidden_size, self.attention_size) * 0.05)
        self.b_omega = nn.Parameter(torch.randn(self.attention_size) * 0.05)
        self.u_omega = nn.Parameter(torch.randn(self.attention_size) * 0.05)

    def forward(self, inputs):

        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        v = torch.tanh(torch.matmul(inputs, self.w_omega) + self.b_omega)
        v = v.view(-1, v.size(1), self.attention_size)

        vu = torch.matmul(v, self.u_omega)
        alphas = torch.softmax(vu, dim=1)
        output = torch.sum(inputs * alphas.unsqueeze(-1), dim=1)

        return output, alphas


###################### ProtBERT Huggingface

class HuggingFace_Transformer(nn.Module):
    def __init__(self, hf_checkpoint, ClassEmbeddingsize):
        super(HuggingFace_Transformer, self).__init__()

        # (1025, 728) # In paper, author mentions W is uniformly distributed
        self.W = torch.nn.Parameter(torch.rand(config.TRANSFORMER_HIDDEN_UNITS + 1, ClassEmbeddingsize + 1) * 0.05)
        self.model = BertModel.from_pretrained(hf_checkpoint)
        
        if config.USE_SECOND_TRANSFORMER:
            for param in self.model.parameters():
                param.requires_grad = False

            self.emb_model = BertModel.from_pretrained(hf_checkpoint)
            self.emb_model_encoder = self.emb_model.encoder
            self.emb_model_pooler = self.emb_model.pooler
        

    def forward(self,X):

        if config.HF_ONLY_ID:
            X_input = X
            attention_mask = None
        else:
            X_input = X['input_ids']
            attention_mask = X['attention_mask']

        # Shape of X should be (batch_size, seq_len)
        X_input = X_input.view(X_input.shape[0], -1)

        if config.USE_SECOND_TRANSFORMER:
            output = self.model(input_ids = X_input, attention_mask = attention_mask)
            output = self.emb_model_encoder(output.last_hidden_state)
            embedding = self.emb_model_pooler(output.last_hidden_state)
            embedding = torch.nn.functional.pad(embedding, (0, 1), value=1)
        else:
            output = self.model(input_ids = X_input, attention_mask = attention_mask) # (batch_size, seq_len, 1024)

            if config.TRANSFORMER_EMBEDDING_TYPE == 'POOLER':
                embedding = torch.nn.functional.pad(output.pooler_output, (0, 1), value=1)
            elif config.TRANSFORMER_EMBEDDING_TYPE == 'CLS':
                embedding = torch.nn.functional.pad(output.last_hidden_state, (0, 1), value=1)
                embedding = embedding[:, 0]
            elif config.TRANSFORMER_EMBEDDING_TYPE == 'ONLY_PHOSPHOSITE':
                embedding = torch.nn.functional.pad(output.last_hidden_state, (0, 1), value=1)
                embedding = embedding[:, int(embedding.shape[1]//2)] # embedding of phosphosite (middle element)
        
        Matmul = torch.matmul(embedding, self.W)

        return Matmul


###################### Transformer (Frozen) -> LSTM

class Transformer_LSTM(nn.Module):
    def __init__(self, vocabnum, seq_lens, ClassEmbeddingsize, hf_checkpoint):
        super(Transformer_LSTM, self).__init__()

        self.transformer_model = BertModel.from_pretrained(hf_checkpoint)
        for param in self.transformer_model.parameters():
                param.requires_grad = False

        self.LSTM = Bi_LSTM(vocabnum, seq_lens, ClassEmbeddingsize)

    def forward(self,X):
        if config.HF_ONLY_ID:
            X_input = X
            attention_mask = None
        else:
            X_input = X['input_ids']
            attention_mask = X['attention_mask']

        # Shape of X should be (batch_size, seq_len)
        X_input = X_input.view(X_input.shape[0], -1)
        batch_embedded = self.transformer_model(input_ids = X_input, attention_mask = attention_mask)
        batch_embedded = batch_embedded.last_hidden_state
        Matmul = self.LSTM(X)
        return Matmul
    

###################### ESM

class ESM(nn.Module):
    def __init__(self, model_name, ClassEmbeddingsize, embedding_mode='avg', is_kinase=False):
        super(ESM, self).__init__()
        
        self.embedding_mode = embedding_mode
        # (1025, 728) # In paper, author mentions W is uniformly distributed
        if not is_kinase:
            self.W = torch.nn.Parameter(torch.rand(get_esm_embedding_dim(model_name) + 1, ClassEmbeddingsize + 1) * 0.05)
        self.esm_model, self.esm_alphabet = load_esm_model(model_name)
        self.last_hidden_state_index = len(self.esm_model.layers) - 1

    def forward(self,X, only_embedding=False):
        # For padding masking
        batch_lens = (X.detach() != self.esm_alphabet.padding_idx).sum(1)

        X = self.esm_model(X, repr_layers=[self.last_hidden_state_index])
        X = X["representations"][self.last_hidden_state_index]

        if self.embedding_mode == 'cls':
            embedding = X[:, 0]
        elif self.embedding_mode == 'avg':
            # Generate per-sequence representations via averaging
            embedding = torch.empty((len(batch_lens), X.size(2)))
            for i, tokens_len in enumerate(batch_lens):
                embedding[i] = X[i, 1:tokens_len - 1].mean(0)

        embedding = torch.nn.functional.pad(embedding, (0, 1), value=1)
        if only_embedding:
            return embedding
        return torch.matmul(embedding, self.W)


class ESM_LSTM(nn.Module):
    def __init__(self, vocabnum, seq_lens, ClassEmbeddingsize, model_name):
        super(ESM_LSTM, self).__init__()
        
        self.esm_model, _ = load_esm_model(model_name)
        self.last_hidden_state_index = len(self.esm_model.layers) - 1
        for param in self.esm_model.parameters():
            param.requires_grad = False

        self.LSTM = Bi_LSTM(vocabnum, seq_lens, ClassEmbeddingsize)

    def forward(self,X):
        X = self.esm_model(X, repr_layers=[self.last_hidden_state_index])
        X = X["representations"][self.last_hidden_state_index]
        return self.LSTM(X)