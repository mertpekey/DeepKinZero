import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time

import config as config

# rnnlib package for Bidirectional Layer Norm LSTM
# https://github.com/daehwannam/pytorch-rnn-library
from rnnlib.seq import LayerNormLSTM, RNNFrame


class Bi_RNN(nn.Module):
    def __init__(self, vocabnum, seq_lens, ClassEmbeddingsize):
        super(Bi_RNN, self).__init__()

        self.vocabnum = vocabnum
        self.seq_lens = seq_lens
        self.ClassEmbeddingsize = ClassEmbeddingsize
        self.num_directions = 2 # Bidirectional

        #self.rnn_cells = [
        #[nn.RNNCell(self.vocabnum, config.NUM_HIDDEN_UNITS),
        #nn.RNNCell(self.vocabnum, config.NUM_HIDDEN_UNITS)],  # 1st bidirectional RNN layer
        #[nn.RNNCell(config.NUM_HIDDEN_UNITS * self.num_directions, config.NUM_HIDDEN_UNITS),
        #nn.RNNCell(config.NUM_HIDDEN_UNITS * self.num_directions, config.NUM_HIDDEN_UNITS)]  # 2nd bidirectional RNN layer
        #]

        # (100, 728) # In paper, author mentions W is uniformly distributed
        self.W = torch.nn.Parameter(torch.rand(config.NUM_HIDDEN_UNITS * 2 + 1, self.ClassEmbeddingsize + 1) * 0.05)
        # Attention
        self.attention = Attention(config.ATTENTION_SIZE, config.NUM_HIDDEN_UNITS * 2)

        self.batchnorm1 = nn.BatchNorm1d(self.vocabnum)
        self.dropout_layer = nn.Dropout1d(p=0.5)
        
        #self.bi_rnn = RNNFrame(self.rnn_cells, dropout=0, bidirectional=True)
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


def softmax(X, theta = 1.0, axis = None):
    """
    Compute the softmax of each element along an axis of X.

    Parameters
    ----------
    X: Tensor Probably should be floats. 
    theta (optional): float parameter, used as a multiplier
        prior to exponentiation. Default = 1.0
    axis (optional): axis to compute values along. Default is the 
        first non-singleton axis.

    Returns an array the same size as X. The result will sum to 1
    along the specified axis.
    """

    X = X.numpy()

    # make X at least 2d
    y = np.atleast_2d(X)

    # find axis
    if axis is None:
        axis = next(j[0] for j in enumerate(y.shape) if j[1] > 1)

    # multiply y against the theta parameter, 
    y = y * float(theta)

    # subtract the max for numerical stability
    y = y - np.expand_dims(np.max(y, axis = axis), axis)
    
    # exponentiate y
    y = np.exp(y)

    # take the sum along the specified axis
    ax_sum = np.expand_dims(np.sum(y, axis = axis), axis)

    # finally: divide elementwise
    p = y / ax_sum

    # flatten if X was 1D
    if len(X.shape) == 1: p = p.flatten()

    return p