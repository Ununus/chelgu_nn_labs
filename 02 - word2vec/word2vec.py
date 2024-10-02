import doc_processor as dproc
from vocabulary import Vocabulary

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm

class TrainInfo():
    def __init__(self) -> None:
        self.n_epoch = 1
        self.batch_size = 1
        self.shuffle = True
        self.learning_rate = 3e-4
        self.device = 'cpu'

class Word2Vec(nn.Module):
    def __init__(self, vocab_size : int, embedding_size : int) -> None:
        super().__init__()
        self.embedding_size = embedding_size
        self.embed = nn.Embedding(vocab_size, embedding_size)
        self.expand_fc = nn.Linear(embedding_size, vocab_size, bias=False)
    def forward(self, center_word_idx):
        x = self.embed(center_word_idx)
        x = self.expand_fc(x)
        x = F.log_softmax(x, dim=-1)
        return x

class Dataset(torch.utils.data.Dataset):
    def __init__(self, docs : dproc.Dataset, vocab : Vocabulary, kernel_size : int = 2) -> None:
        '''Для каждого i-го слова в документах docs выделяем контекст [i - kernel_size; i + kernel_size]
            слова не из словаря vocab игнорируются
        '''
        super().__init__()
        self.data_ = list()
        self.docs = docs
        self.vocab = vocab
        print('Создание датасета для тренировки word2vec со скользящим окном размера {}'.format(kernel_size))
        for doc, _ in docs.get_train():
            for idx, word in enumerate(doc):
                if not vocab.has_word(word):
                    continue
                word_ix = vocab.word_to_index(word)
                w_idx = idx - kernel_size
                if w_idx < 0:
                    w_idx = 0
                while w_idx < len(doc) and w_idx <= idx + kernel_size:
                    ctx_word = doc[w_idx]
                    if w_idx == idx or not vocab.has_word(ctx_word):
                        w_idx += 1
                        continue
                    ctx_word_ix = vocab.word_to_index(ctx_word)
                    self.data_.append((word_ix, ctx_word_ix))
                    w_idx += 1
        print('Размер word2vec датасета:', len(self.data_))

    def __len__(self):
        return len(self.data_)
    
    def __getitem__(self, idx):
        return self.data_[idx]
    
def train_word2vec(net : Word2Vec, dataset : Dataset, info : TrainInfo):
    print('Тренировка сети wrod2vec...')
    net = net.to(info.device)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=info.batch_size, shuffle=info.shuffle)
    loss_function = nn.NLLLoss(reduction='sum')
    optimizer = optim.AdamW(net.parameters(), lr=info.learning_rate)
    running_loss = []
    net.train()
    for epoch in range(info.n_epoch):
        print("Epoch {}/{}".format(epoch+1, info.n_epoch))
        epoch_loss = 0
        for center_ix, contex_ix in tqdm(dataloader):
            center_ix, contex_ix = center_ix.to(info.device), contex_ix.to(info.device)
            optimizer.zero_grad()
            log_probs = net(center_ix)
            loss = loss_function(log_probs, contex_ix)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.cpu().item()
        epoch_loss /= len(dataset)
        running_loss.append(epoch_loss)
        print("\tLoss: {:.4f}".format(epoch_loss))
    return running_loss

