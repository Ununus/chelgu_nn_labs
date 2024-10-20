import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class DocClassifierCreateInfo:
    def __init__(self) -> None:
        self.embedding_size : int = 100
        self.num_classes : int = 2
        self.hidden_layer_size : int = 128

class DocClassifier(nn.Module):
    def __init__(self, info : DocClassifierCreateInfo) -> None:
        super().__init__()
        self.linear1 = nn.Linear(info.embedding_size, info.hidden_layer_size)
        self.linear2 = nn.Linear(info.hidden_layer_size, info.num_classes, bias=False)
    def forward(self, x):
        x = self.linear1(x)
        x = F.relu(x)
        x = self.linear2(x)
        x = F.log_softmax(x, dim=-1)
        return x
