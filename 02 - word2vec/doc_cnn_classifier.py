import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class DocCnnClassifierCreateInfo:
    def __init__(self) -> None:
        self.embedding_size : int = 100
        self.num_classes : int = 2
        self.hidden_layer_size : int = 128
        self.num_filters : int = 100

class DocClassifier(nn.Module):
    def __init__(self, info : DocCnnClassifierCreateInfo) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(info.embedding_size, info.num_filters, 3, 1, 1)
        self.conv2 = nn.Conv1d(info.embedding_size, info.num_filters, 5, 1, 2)
        self.conv3 = nn.Conv1d(info.embedding_size, info.num_filters, 7, 1, 3)
        self.linear1 = nn.Linear(info.num_filters * 3, info.hidden_layer_size)
        self.linear2 = nn.Linear(info.hidden_layer_size, info.num_classes, bias=False)
    def forward(self, x):
        t1 = self.conv1(x)
        t2 = self.conv2(x)
        t3 = self.conv3(x)
        t1 = F.relu(t1)
        t2 = F.relu(t2)
        t3 = F.relu(t3)
        t1 = F.max_pool1d(t1, x.shape[2], x.shape[2])
        t2 = F.max_pool1d(t2, x.shape[2], x.shape[2])
        t3 = F.max_pool1d(t3, x.shape[2], x.shape[2])
        t1 = t1.squeeze(dim=2)
        t2 = t2.squeeze(dim=2)
        t3 = t3.squeeze(dim=2)
        x = torch.cat((t1, t2, t3), dim=1)
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=-1)
        return x