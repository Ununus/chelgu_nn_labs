import doc_processor as dproc
from vocabulary import Vocabulary
from word2vec import Word2Vec
import torch
from tqdm import tqdm

class DocumentsEmbeddingSumWord():
    def __init__(self, docs, vocab : Vocabulary, word2vec : Word2Vec) -> None:
        '''Для каждого документа docs вычислить эмбеддинг как сумму эмбеддингов его слов,
            если они есть в словаре vocab
        '''
        self.data = list() # list<torch.tensor>
        self.docs = docs
        self.vocab = vocab
        print('Создание эмбеддингов документов...')
        word2vec.eval()
        with torch.no_grad():
            for doc, _ in tqdm(docs):
                emned_vec_sum = torch.zeros_like(word2vec.embed.weight[0])
                for word in doc:
                    if vocab.has_word(word):
                        word_ix = vocab.word_to_index(word)
                        word_emned_vec = word2vec.embed.weight[word_ix]
                        emned_vec_sum += word_emned_vec
                self.data.append(emned_vec_sum)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]