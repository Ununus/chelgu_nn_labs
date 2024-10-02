from tqdm import tqdm
from collections import Counter
import datasets
import string
import nltk #Библиотека токенизации https://www.nltk.org/ , ещё есть библиотека razdel для русского языка
import pymorphy3 #для лемматизации слов, ещё есть PyMystem3 и RNNMorphPredictor
import re
import random
import os

class DocumentList:
    '''Набор документов и их обработка'''
    def __init__(self) -> None:
        self.raw_data = list()
        self.data = list() # list<list<str>, int> (list<list<token_str>, group_ix>)
        self.groups = []
        self.group2ix = {}

    def add_document(self, document : str, group : str) -> None:
        if group not in self.group2ix:
            self.group2ix[group] = len(self.groups)
            self.groups.append(group)
        self.raw_data.append((document, self.group2ix[group]))

    def process(self, to_lower : bool = True, delete_punct : bool = True, delete_stopwords : bool = True, delete_numbers = True) -> None:
        self.init_nltk_()
        print('to_lower:', to_lower)
        print('delete_punct:', delete_punct)
        print('delete_stopwords:', delete_stopwords)
        print('delete_numbers:', delete_numbers)
        print('Обработка документов...')
        if delete_stopwords:
            stop_words = set(nltk.corpus.stopwords.words('russian'))
        morph = pymorphy3.MorphAnalyzer(lang='ru')
        for (text, group_ix) in tqdm(self.raw_data):
            if to_lower:
                text = text.lower()
            if delete_numbers:
                text = re.sub(r'\d+', '', text)
            tokens = nltk.tokenize.word_tokenize(text)
            new_tokens = []
            for token in tokens:
                if delete_punct and token in string.punctuation:
                    continue
                lem = morph.parse(token)
                word = lem[0].normal_form
                if delete_stopwords and word in stop_words:
                    continue
                new_tokens.append(word)
            self.data.append((new_tokens, group_ix))

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]

    def init_nltk_(self):
        '''Загрузим токенизер для русского языка'''
        print('Загрузка токенизера...')
        nltk.data.path = os.curdir
        if not os.path.exists(os.path.join(os.curdir, 'tokenizers')):
            nltk.download('punkt', os.curdir)
            nltk.download('punkt_tab', os.curdir)
            nltk.download('stopwords', os.curdir)
        nltk.data.load('nltk:tokenizers/punkt/russian.pickle')

class DocumentListProxy:
    def __init__(self, docs : DocumentList) -> None:
        self.docs = docs
        self.ixs = list()
        self.group_cnt = Counter()
    def add_document(self, index):
        self.ixs.append(index)
        self.group_cnt[self.docs[index][1]] += 1
    def __len__(self):
        return len(self.ixs)
    def __getitem__(self, idx):
        return self.docs[self.ixs[idx]]

class Dataset:
    def __init__(self, docs : DocumentList, num_in_train : int, num_in_validation : int, num_in_test : int, skip_empty : bool = True) -> None:
        num_in_train, num_in_validation, num_in_test = num_in_train // 2, num_in_validation // 2, num_in_test // 2
        print('Разбиение документов на обучающее, валидационное и тестовое множества')
        self.train_data = DocumentListProxy(docs)
        self.validation_data = DocumentListProxy(docs)
        self.test_data = DocumentListProxy(docs)
        cnt = Counter()
        skipped_cnt = 0
        for ix in range(len(docs)):
            if skip_empty and len(docs[ix][0]) == 0:
                skipped_cnt += 1
                continue
            group = docs[ix][1]
            if cnt[(0, group)] < num_in_test:
                self.test_data.add_document(ix)
                cnt[(0, group)] += 1
            elif cnt[(1, group)] < num_in_validation:
                self.validation_data.add_document(ix)
                cnt[(1, group)] += 1
            elif num_in_train == -1 or cnt[(2, group)] < num_in_train:
                self.train_data.add_document(ix)
                cnt[(2, group)] += 1
        if skip_empty:
            print('Пропущенно пустых документов:', skipped_cnt)
        self.print_('Обучающее множество:', self.train_data)
        self.print_('Валидационное множество:', self.validation_data)
        self.print_('Тестовое множество:', self.test_data)
    def print_(self, split, data):
        print(split, len(data))
        for group_ix, cnt in data.group_cnt.items():
            print('\t{} {}: {}'.format(group_ix, data.docs.groups[group_ix], cnt))
    def get_train(self):
        return self.train_data
    def get_validation(self):
        return self.validation_data
    def get_test(self):
        return self.test_data

def load_russian_youtube_comments_dataset(limit_number_of_documents : int = None):
    limit_number_of_documents_in_class = limit_number_of_documents // 2
    print('Загрузка датасета из ютуберских комментов...')
    ds = datasets.load_dataset('Maxstan/russian_youtube_comments_political_and_nonpolitical')
    if not limit_number_of_documents_in_class is None:
        print('Ограничение на количество документов в группе:', limit_number_of_documents_in_class)
    print('Выборка из датасета...')
    ds = ds['train'].to_list()
    random.shuffle(ds)
    docs = DocumentList()
    group_cnt = Counter()
    docs.groups = ['Фоновые', 'Политические']
    docs.group2ix = {'Фоновые' : 0, 'Политические' : 1}
    for dc in tqdm(ds):
        text, group = dc['text'], dc['group']
        if not limit_number_of_documents_in_class is None and group_cnt[group] >= limit_number_of_documents_in_class:
            continue
        docs.add_document(text, group)
        group_cnt[group] += 1
    print('Загружено из датасета:')
    for group, cnt in group_cnt.items():
        print('\t{} {}: {}'.format(docs.group2ix[group], group, cnt))
    return docs

def load_custom_dataset() -> DocumentList:
    print('Загрузка тестового ...')
    docs = DocumentList()
    docs.add_document('w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14', 'class_1')
    docs.add_document('w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20', 'class_2')
    return docs

def load_custom_dataset_2(num_docs_in_class : int = 100, min_doc_length : int = 4, max_doc_length : int = 40) -> DocumentList:
    '''w1, w2, w3, w4, w5 уникальны для class_1
        w15 w16 w17 w18 w19 w20' уникальны для 'class_2',
        остальные встречаются и там и там
    '''
    print('Загрузка тестового ...')
    docs = DocumentList()
    class_1 = 'w1 w2 w3 w4 w5 w6 w7 w8 w9 w10 w11 w12 w13 w14'.split()
    class_2 = 'w6 w7 w8 w9 w10 w11 w12 w13 w14 w15 w16 w17 w18 w19 w20'.split()
    def gen_doc(class_words):
        doc_len = random.randint(min_doc_length, max_doc_length)
        word_list = []
        for i in range(doc_len):
            word_ix = random.randint(0, len(class_1) - 1)
            word_list.append(class_words[word_ix])
        doc = ' '.join(x for x in word_list)
        return doc
    for _ in range(num_docs_in_class):
        docs.add_document(gen_doc(class_1), 'class_1')
        docs.add_document(gen_doc(class_2), 'class_2')
    return docs

    
        