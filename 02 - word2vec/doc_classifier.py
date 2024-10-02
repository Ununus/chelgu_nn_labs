from doc_embedding import DocumentsEmbeddingSumWord

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

class TrainInfo():
    def __init__(self) -> None:
        self.n_epoch = 1
        self.batch_size = 1
        self.shuffle = False
        self.learning_rate = 3e-4
        self.device = 'cpu'

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
    
class Dataset(torch.utils.data.Dataset):
    def __init__(self, doc_embeds : DocumentsEmbeddingSumWord):
        self.doc_embeds = doc_embeds

    def __len__(self):
        return len(self.doc_embeds)
    
    def __getitem__(self, idx):
        return self.doc_embeds[idx], self.doc_embeds.docs[idx][1]
        
def train_one_epoch(net, train_dataloader, optimizer, device):
    net.train()
    epoch_loss_sum = 0.
    for doc_embed, doc_group in tqdm(train_dataloader):
        doc_embed, doc_group = doc_embed.to(device), doc_group.to(device)
        optimizer.zero_grad()
        log_probs = net(doc_embed)
        loss = F.nll_loss(log_probs, doc_group, reduction='sum')
        loss.backward()
        optimizer.step()
        epoch_loss_sum += loss.cpu().item()
    return epoch_loss_sum

def test(net, test_dataloader, device):
    net.eval()
    test_loss_sum = 0.
    correct_sum = 0
    with torch.no_grad():
        for doc_embed, doc_group in test_dataloader:
            doc_embed, doc_group = doc_embed.to(device), doc_group.to(device)
            log_probs = net(doc_embed)
            loss = F.nll_loss(log_probs, doc_group, reduction='sum')
            pred = log_probs.argmax(dim=-1, keepdim=True)
            test_loss_sum += loss.cpu().item()
            correct_sum += pred.eq(doc_group.view_as(pred)).sum().cpu().item()
    return test_loss_sum, correct_sum

def train_network(net : DocClassifier, train_dataset : Dataset, validation_dataset : Dataset, info : TrainInfo):
    print('Тренировка сети DocClassifier...')
    net = net.to(info.device)
    best_loss, best_loss_last_ix = 0., -1
    do_validation = not validation_dataset is None
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=info.batch_size, shuffle=info.shuffle)
    train_losses = []
    if do_validation:
        validation_dataloader = torch.utils.data.DataLoader(validation_dataset, batch_size=info.batch_size, shuffle=False)
        validation_losses, validation_accuraces = [], []
    optimizer = optim.Adadelta(net.parameters(), lr=info.learning_rate)
    for epoch in range(info.n_epoch):
        print("Epoch {}/{}".format(epoch+1, info.n_epoch))
        train_loss_sum = train_one_epoch(net, train_dataloader, optimizer, info.device)
        train_loss = train_loss_sum / len(train_dataset)
        train_losses.append(train_loss)
        print('\tLoss: {:.4f}'.format(train_loss))
        if do_validation:
            validation_loss_sum, correct_sum = test(net, validation_dataloader, info.device)
            validation_loss = validation_loss_sum / len(validation_dataset)
            accuracy = correct_sum * 100. / len(validation_dataset)
            validation_losses.append(validation_loss)
            validation_accuraces.append(accuracy)
            if best_loss_last_ix == -1 or validation_loss + 1e-4 < best_loss:
                best_loss_last_ix = epoch
                best_loss = validation_loss
            print('\nTest: Loss: {:.4f}, Accuracy {}/{} ({:.2f}%), Best loss: {:.4f}\n'.format(
                validation_loss, correct_sum, len(validation_dataset), accuracy, best_loss))
            if best_loss_last_ix + 5 <= epoch:
                print('Loss не улучшается больше 5 эпох. Остановка обучения.')
                break
    if do_validation:
        return train_losses, validation_losses
    return train_losses
            