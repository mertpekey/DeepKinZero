import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size*4) # 4 because LSTM has 4 gates
    
    def forward(self, input, hx):
        h, c = hx
        gates = self.lstm_cell(input, (h, c))
        normalized_gates = self.layer_norm(gates)
        new_h, new_c = nn.functional.lstm_cell(input, normalized_gates)
        return new_h, new_c


class MyModel(nn.Module):
    def __init__(self, Params, vocabnum, seq_lens, ClassEmbeddingsize):
        super(MyModel, self).__init__()

        self.Params = Params
        self.vocabnum = vocabnum
        self.seq_lens = seq_lens
        self.ClassEmbeddingsize = ClassEmbeddingsize


        self.W = torch.nn.Parameter(torch.randn(DE.shape[1], self.ClassEmbeddingsize + 1) * 0.05)

        self.batchnorm1 = nn.BatchNorm1d(self.batch_ph.size(1))
        self.dropout_layer = nn.Dropout(p=0.5)
        
        fw_cells = [LayerNormLSTMCell(self.Params["num_hidden_units"], self.Params["num_hidden_units"]) for L in range(self.Params["num_layers"])]
        bw_cells = [LayerNormLSTMCell(self.Params["num_hidden_units"], self.Params["num_hidden_units"]) for L in range(self.Params["num_layers"])]

        self.rnn_dummy = LayerNormLSTMCell(input_size, hidden_size)
        rnn_outputs, _ = torch.nn.utils.rnn.bidirectional_dynamic_rnn(
                            fw_cells, bw_cells, batch_embedded, 
                            sequence_length=self.seq_len_ph[GraphID], 
                            dtype=torch.float32)

        self.batchnorm2 = nn.BatchNorm1d(rnn_outputs.size()[1])

    def forward(self,batch_embedded):
        x = self.batchnorm1(batch_embedded)
        x = self.dropout_layer(x)
        x = self.rnn_dummy(x)
        x = self.batchnorm2(x)
        x, alphas = self.attention(x, self.Params["ATTENTION_SIZE"], GraphID, return_alphas=True)
        x = self.dropout_layer(x)
        embedding = torch.nn.functional.pad(x, (0, 1), value=1)


    def attention(self, inputs, attention_size, GraphID, time_major=False, return_alphas=False):
        
        if isinstance(inputs, tuple):
            # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
            inputs = torch.cat(inputs, 2)
        
        if time_major:
            # (T,B,D) => (B,T,D)
            inputs = inputs.permute(1,0,2)
        
        hidden_size = inputs.shape[2]  # D value - hidden size of the RNN layer
        
        # Trainable parameters
        w_omega = torch.randn(hidden_size, attention_size) * 0.05
        b_omega = torch.randn(attention_size) * 0.05
        u_omega = torch.randn(attention_size) * 0.05
        
        if torch.cuda.is_available():
            w_omega = w_omega.cuda()
            b_omega = b_omega.cuda()
            u_omega = u_omega.cuda()
        
        with torch.no_grad():
            # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
            # the shape of `v` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
            v = torch.tanh(torch.einsum('ijk,kl->ijl', inputs, w_omega) + b_omega)
            v = v.view(inputs.shape[0], inputs.shape[1], attention_size)
        
        # For each of the timestamps its vector of size A from `v` is reduced with `u` vector
        vu = torch.einsum('ijk,k->ij', v, u_omega)  # (B,T) shape
        alphas = F.softmax(vu, dim=1)  # (B,T) shape
        
        # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
        output = torch.einsum('ijk,ij->ik', inputs, alphas)
        
        if not return_alphas:
            return output
        else:
            return output, alphas

    def softmax(self, X, theta = 1.0, axis = None):
        """
        Compute the softmax of each element along an axis of X.
    
        Parameters
        ----------
        X: ND-Array. Probably should be floats. 
        theta (optional): float parameter, used as a multiplier
            prior to exponentiation. Default = 1.0
        axis (optional): axis to compute values along. Default is the 
            first non-singleton axis.
    
        Returns an array the same size as X. The result will sum to 1
        along the specified axis.
        """
    
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