import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class EndToEndModel(nn.Module):
    def __init__(self, vocabnum, class_embedding_size, params=None, seqlens=15, seed=None, write_embedding_vis=False):
        super(EndToEndModel, self).__init__()
        
        if params is not None:
            self.params = params
        self.seed = seed
        torch.manual_seed(self.seed)
        self.vocabnum = vocabnum
        self.seq_lens = seqlens
        self.ModelSaved = False
        self.class_embedding_size = class_embedding_size
        self.write_embedding_vis = write_embedding_vis
        self.Models = []
        self.Graphs = []
        self.batch_ph = []
        self.seq_len_ph = []
        self.keep_prob_ph = []
        self.is_training = []
        self.CE = []
        self.CKE = []
        self.TCI = []
        self.alphas = []
        self.train_writer = {}
        self.test_writer = {}
        self.val_writer = {}
        self.init_params()
        self.initilize()

    def init_params(self):
        self.params = {
            "rnn_unit_type": "LNlstm", 
            "num_layers": 1, 
            "num_hidden_units": 500, 
            "dropoutval": 0.5, 
            "learningrate": 0.003, 
            "useAtt": False, 
            "useEmbeddingLayer": False, 
            "useEmbeddingLayer": False, 
            "num_of_Convs": [], 
            "UseBatchNormalization1": True, 
            "UseBatchNormalization2": True, 
            "EMBEDDING_DIM": 500, 
            "ATTENTION_SIZE": 5, 
            "IncreaseEmbSize": 0, 
            "Bidirectional": True, 
            "Dropout1": True, 
            "Dropout2": True, 
            "Dropout3": False, 
            "regs": 0.001, 
            "batch_size": 64, 
            "ClippingGradients": 9.0, 
            "activation1": "tanh", 
            "LRDecay": True, 
            "seed": None, 
            "NumofModels": 3
        }
        self.LogDir = "" 
        self.ckpt_dir = "" 
        self.loss = []
        self.num_examples = 0 
        self.training_epochs = 200 
        self.display_step = 10 
        self.DELTA = 0.5 
        self.useEmbeddingLayer = True 
        self.IncreaseEmbSize = 0 

    def rnn_cell(self, L=0):
        # Get the cell type
        if self.Params["rnn_unit_type"] == 'rnn':
            rnn_cell_type = nn.RNNCell
        elif self.Params["rnn_unit_type"] == 'gru':
            rnn_cell_type = nn.GRUCell
        elif self.Params["rnn_unit_type"] == 'lstm':
            rnn_cell_type = nn.LSTMCell
        elif self.Params["rnn_unit_type"] == 'LNlstm':
            raise NotImplementedError("LayerNormBasicLSTMCell not available in PyTorch")
        elif self.Params["rnn_unit_type"] == 'CUDNNLSTM':
            raise NotImplementedError("CudnnLSTM not available in PyTorch")
        else:
            raise Exception("Choose a valid RNN unit type.")

        #Create a layer
        if L == self.Params["num_layers"] - 1:
            single_cell = rnn_cell_type(self.Params["input_size"], self.Params["num_hidden_units"], bias=True, nonlinearity='linear')
        else:
            if self.Params["activation1"] == "None":
                single_cell = rnn_cell_type(self.Params["input_size"], self.Params["num_hidden_units"], bias=True, nonlinearity='linear')
            else:
                ## Nonlinearity ne yapilacak ???
                single_cell = rnn_cell_type(self.Params["input_size"], self.Params["num_hidden_units"], bias=True)

        return single_cell


    def _add_conv_layers(self, inputs):
        """Adds convolution layers."""
        convolved = inputs
        for i in range(len(self.Params["num_of_Convs"])):
            convolved_input = convolved
            if self.Params["UseBatchNormalization1"]:
                convolved_input = nn.BatchNorm1d(convolved_input.size(1)).to(convolved_input.device)(convolved_input)
            # Add dropout layer if enabled and not first convolution layer.
            if i > 0 and (self.Params["Dropout1"]):
                convolved_input = nn.Dropout(self.Params["dropoutval"])(convolved_input)
            convolved = nn.Conv1d(
                in_channels=convolved_input.size(1),
                out_channels=self.Params["num_of_Convs"][i],
                kernel_size=5,
                stride=1,
                padding=2,
                bias=True)(convolved_input)
            convolved = F.relu(convolved)
        return convolved





    ######## I am not sure where these come from
    def create_graph(self, i):
        model = nn.Sequential(
            nn.Linear(self.vocabnum, self.params["EMBEDDING_DIM"]),
            nn.ReLU(),
            nn.LSTM(input_size=self.params["EMBEDDING_DIM"], hidden_size=self.params["num_hidden_units"],
                    num_layers=self.params["num_layers"], dropout=self.params["dropoutval"],
                    bidirectional=self.params["Bidirectional"]),
            nn.Linear(self.params["num_hidden_units"], self.ClassEmbeddingsize),
            nn.Dropout(self.params["dropoutval"]),
            nn.Linear(self.ClassEmbeddingsize, self.params["classnumber"])
        )
        return model

    def initilize(self):
        for i in range(self.params["NumofModels"]):
            self.Models[i].train()
            self.optimizer = optim
