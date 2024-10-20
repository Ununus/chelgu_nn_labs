import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class DocRnnClassifierCreateInfo:
    def __init__(self) -> None:
        self.embedding_size : int = 100
        self.num_classes : int = 2
        self.hidden_layer_size : int = 128
        self.rnn_type = 'gru' # rnn, gru, lstm
        self.aggregation_type = 'max' # mean, max

class DocRnnClassifier(nn.Module):
    def __init__(self, info : DocRnnClassifierCreateInfo) -> None:
        super().__init__()
        if info.rnn_type == 'rnn':
            self.rnn = nn.RNN(info.embedding_size, info.hidden_layer_size, batch_first=True)
        elif info.rnn_type == 'gru':
            self.rnn = nn.GRU(info.embedding_size, info.hidden_layer_size, batch_first=True)
        elif info.rnn_type == 'lstm':
            self.rnn = nn.LSTM(info.embedding_size, info.hidden_layer_size, batch_first=True)
        else:
            raise ValueError('Invalid rnn_type')
        self.linear1 = nn.Linear(info.hidden_layer_size, info.hidden_layer_size)
        self.linear2 = nn.Linear(info.hidden_layer_size, info.num_classes, bias=False)
        self.aggregation_type = info.aggregation_type
    def forward(self, x):
        x = x.permute(0, 2, 1) #(batch, seq, feature)
        x, _ = self.rnn(x)
        if self.aggregation_type == 'max':
            x = x.max(dim=1)[0]
        elif self.aggregation_type == 'mean':
            x = x.mean(dim=1)[0]
        else:
            raise ValueError('Invalid aggragation_type')
        x = F.tanh(x)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=-1)
        return x
