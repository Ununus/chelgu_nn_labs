#pip install tqdm torch torchvision torchaudio datasets nltk pymorphy3
import doc_processor as dproc
from vocabulary import Vocabulary
import word2vec as w2v
from doc_embedding import DocumentsEmbeddingSumWord
import doc_classifier as dcl

import os
import pickle
import torch
import json

word_to_vec_path = '02 - word2vec'

class PickleInfo:
    def __init__(self) -> None:
        self.load_processed_documents = True
        self.save_processed_documents = True
        self.load_vocabulary = True
        self.save_vocabulary = True
        self.load_word2vec = True
        self.save_word2vec = True
        self.load_classifier = True
        self.save_classifier = True
        self.data_path = os.path.join(os.curdir, word_to_vec_path, 'data') #Сюда будем сувать pickle
        if not os.path.exists(self.data_path):
            os.mkdir(self.data_path)
        self.processed_documents_filename = os.path.join(self.data_path, 'processed_documents.pickle')
        self.vocabulary_filename = os.path.join(self.data_path, 'vocabulary.pickle')
        self.word2vec_filename = os.path.join(self.data_path, 'word2vec.pth')
        self.classifier_filename = os.path.join(self.data_path, 'classifier.pth')

def load_processed_documents(info : PickleInfo):
    print('Загрузка обработанных документов ({})'.format(info.processed_documents_filename))
    with open(info.processed_documents_filename, 'rb') as file:
        docs = pickle.load(file)
    return docs
def save_processed_documents(info : PickleInfo, docs : dproc.DocumentList):
    print('Сохранение обработанных документов ({})'.format(info.processed_documents_filename))
    with open(info.processed_documents_filename, 'wb') as file:
        pickle.dump(docs, file)

def load_vocabulary(info : PickleInfo):
    print('Загрузка словаря ({})'.format(info.vocabulary_filename))
    with open(info.vocabulary_filename, 'rb') as file:
        vocab = pickle.load(file)
    return vocab
def save_vocabulary(info : PickleInfo, vocab : Vocabulary):
    print('Сохранение словаря ({})'.format(info.vocabulary_filename))
    with open(info.vocabulary_filename, 'wb') as file:
        pickle.dump(vocab, file)

def load_pretrained_word2vec(info : PickleInfo):
    print('Загрузка обученной word2vec ({})'.format(info.word2vec_filename))
    net = torch.load(info.word2vec_filename)
    return net
def save_pretrained_word2vec(info : PickleInfo, net : w2v.Word2Vec):
    print('Сохранение обученной word2vec ({})'.format(info.word2vec_filename))
    torch.save(net, info.word2vec_filename)

def load_pretrained_classifier(info : PickleInfo):
    print('Загрузка обученной classifier ({})'.format(info.word2vec_filename))
    net = torch.load(info.classifier_filename)
    return net
def save_pretrained_classifier(info : PickleInfo, net : dcl.DocClassifier):
    print('Сохранение обученной classifier ({})'.format(info.word2vec_filename))
    torch.save(net, info.classifier_filename)

def main(my_params):
    info = PickleInfo()
    info.load_processed_documents = my_params['Load processed documents']
    info.save_processed_documents = my_params['Save processed documents']
    info.processed_documents_filename = os.path.join(info.data_path, my_params['Processed documents filename'])
    info.load_vocabulary = my_params['Load vocabulary']
    info.save_vocabulary = my_params['Save vocabulary']
    info.vocabulary_filename = os.path.join(info.data_path, my_params['Vocabulary filename'])
    info.load_word2vec = my_params['Load word2vec']
    info.save_word2vec = my_params['Save word2vec']
    info.word2vec_filename = os.path.join(info.data_path, my_params['Word2vec filename'])
    info.load_classifier = my_params['Load classifier']
    info.save_classifier = my_params['Save classifier']
    info.classifier_filename = os.path.join(info.data_path, my_params['Classifier filename'])
    train_dataset_size = my_params['Train dataset size']
    validation_dataset_size = my_params['Validation dataset size']
    test_dataset_size = 100
    number_of_documents_to_process = int((train_dataset_size + validation_dataset_size + test_dataset_size) * 1.1)
    vocabulary_word_count_threshold = my_params['Vocabulary word count threshold']
    train_device = my_params['Train device']
    word2vec_kernel_size = my_params['Word2vec kernel size']
    word2vec_embedding_size = my_params['Word2vec embedding size']
    word2vec_train_number_of_epochs = my_params['Word2vec train number of epochs']
    word2vec_train_batch_size = my_params['Word2vec train batch size']
    word2vec_train_learning_rate = my_params['Word2vec train learning rate']
    classifier_hidden_layer_size = my_params['Classifier hidden layer size']
    classifier_train_number_of_epochs = my_params['Classifier train number of epochs']
    classifier_train_batch_size = my_params['Classifier train batch size']
    classifier_train_learning_rate = my_params['Classifier train learning rate']
    inference_filename = os.path.join(word_to_vec_path, my_params['Inference filename'])
    # Документы
    if info.load_processed_documents and os.path.exists(info.processed_documents_filename):
        documents = load_processed_documents(info)
    else:
        documents = dproc.load_russian_youtube_comments_dataset(number_of_documents_to_process)
        documents.process(to_lower=True, delete_punct=True, delete_stopwords=True, delete_numbers=True)
        if info.save_processed_documents:
            save_processed_documents(info, documents)
    dc_dataset = dproc.Dataset(documents, train_dataset_size, validation_dataset_size, test_dataset_size)
    # Словарь
    if info.load_vocabulary and os.path.exists(info.vocabulary_filename):
        vocab = load_vocabulary(info)
    else:
        vocab = Vocabulary(dc_dataset, word_count_threshold=vocabulary_word_count_threshold)
        if info.save_vocabulary:
            save_vocabulary(info, vocab)
    # word2vec
    if info.load_word2vec and os.path.exists(info.word2vec_filename):
        word2vec = load_pretrained_word2vec(info)
    else:
        word2vec_dataset = w2v.Dataset(dc_dataset, vocab, kernel_size=word2vec_kernel_size)
        word2vec = w2v.Word2Vec(len(vocab), embedding_size=word2vec_embedding_size)
        word2vec_train_info = w2v.TrainInfo()
        word2vec_train_info.n_epoch = word2vec_train_number_of_epochs
        word2vec_train_info.batch_size = word2vec_train_batch_size
        word2vec_train_info.learning_rate = word2vec_train_learning_rate
        word2vec_train_info.shuffle = True
        word2vec_train_info.device = train_device
        w2v.train_word2vec(word2vec, word2vec_dataset, word2vec_train_info)
        if info.save_word2vec:
            save_pretrained_word2vec(info, word2vec)
    # classifier
    if info.load_classifier and os.path.exists(info.classifier_filename):
        doc_classifier_net = load_pretrained_classifier(info)
    else:
        train_doc_embeds = DocumentsEmbeddingSumWord(dc_dataset.get_train(), vocab, word2vec)
        word2vec = word2vec.cpu()
        validation_doc_embeds = DocumentsEmbeddingSumWord(dc_dataset.get_validation(), vocab, word2vec)
        test_doc_embeds = DocumentsEmbeddingSumWord(dc_dataset.get_test(), vocab, word2vec)
        doc_classifier_train_dataset = dcl.Dataset(train_doc_embeds)
        doc_classifier_validation_dataset = dcl.Dataset(validation_doc_embeds)
        doc_classifier_test_dataset = dcl.Dataset(test_doc_embeds)
        doc_classifier_create_info = dcl.DocClassifierCreateInfo()
        doc_classifier_create_info.embedding_size = word2vec.embedding_size
        doc_classifier_create_info.hidden_layer_size = classifier_hidden_layer_size
        doc_classifier_create_info.num_classes = 2
        doc_classifier_net = dcl.DocClassifier(doc_classifier_create_info)
        doc_classifier_train_info = dcl.TrainInfo()
        doc_classifier_train_info.n_epoch = classifier_train_number_of_epochs
        doc_classifier_train_info.batch_size = classifier_train_batch_size
        doc_classifier_train_info.learning_rate = classifier_train_learning_rate
        doc_classifier_train_info.shuffle = True
        doc_classifier_train_info.device = train_device
        dcl.train_network(doc_classifier_net, doc_classifier_train_dataset, doc_classifier_validation_dataset, doc_classifier_train_info)
        if info.save_classifier:
            save_pretrained_classifier(info, doc_classifier_net)
        print('Тест классификатора')
        word2vec = word2vec.cpu()
        doc_classifier_net = doc_classifier_net.cpu()
        doc_classifier_net.eval()
        with torch.no_grad():
            for idx, (doc_embed, group_ix) in enumerate(doc_classifier_test_dataset):
                logits = doc_classifier_net(doc_embed)
                pred = logits.argmax(dim=-1)
                #print(' '.join(dc_dataset.get_test()[idx][0]))
                tok_color_beg = '\033[34m'
                tok_color_end = '\033[0m'
                num_in_vocab = 0
                for word in dc_dataset.get_test()[idx][0]:
                    if vocab.has_word(word):
                        print('{}'.format(word), end=' ')
                        num_in_vocab += 1
                    else:
                        print('{}{}{}'.format(tok_color_beg, word, tok_color_end), end=' ')
                if pred.item() == group_ix:
                    tok_color_beg = '\033[32m'
                else:
                    tok_color_beg = '\033[31m'
                # (Предсказанный, Правильный)
                print('[{}({}, true={}){}, слов={}]'.format(tok_color_beg, documents.groups[pred.item()], documents.groups[group_ix], tok_color_end, num_in_vocab))
    # inference
    if os.path.exists(inference_filename):
        print('Запуск на отдельных примерах из файла', inference_filename)
        custom_political = dproc.DocumentList()
        custom_political.groups = ['Фоновые', 'Политические']
        custom_political.group2ix = {'Фоновые' : 0, 'Политические' : 1}
        with open(inference_filename, 'r',encoding='utf-8') as file:
            doc = file.readline().strip()
            while doc:
                custom_political.add_document(doc, 'Политические')
                doc = file.readline().strip()
        custom_political.process()
        custom_embeds = DocumentsEmbeddingSumWord(custom_political, vocab, word2vec)
        with torch.no_grad():
            for idx, doc_embed in enumerate(custom_embeds):
                logits = doc_classifier_net(doc_embed)
                pred = logits.argmax(dim=-1)
                print('\033[0;33m', custom_political.raw_data[idx][0], '\033[0m', end=' ')
                #print(custom_political[idx][0])
                if pred.item() == 0:
                    print('\t\033[1;31m', custom_political.groups[pred.item()], '\033[0m')
                else:
                    print('\t\033[1;32m', custom_political.groups[pred.item()], '\033[0m')
    else:
        print('Не могу найти путь:', inference_filename)

if __name__ == '__main__':
    if os.path.abspath(os.curdir).endswith(word_to_vec_path):
        word_to_vec_path = '.'
    params_path = os.path.join(os.curdir, word_to_vec_path, 'params.json')
    if os.path.exists(params_path):
        with open(params_path, 'r', encoding='utf-8') as file:
            my_params = json.load(file)
    else:
        my_params = {
            'Load processed documents': True, 
            'Save processed documents' : True,
            'Processed documents filename': 'processed_documents.pickle',
            'Load vocabulary': True, 
            'Save vocabulary': True,
            'Vocabulary filename': 'vocabulary.pickle',
            'Load word2vec': True,
            'Save word2vec': True,
            'Word2vec filename': 'word2vec.pth',
            'Load classifier': True,
            'Save classifier': True,
            'Classifier filename': 'classifier.pth',
            'Train dataset size': 100000,
            'Validation dataset size': 20000,
            'Vocabulary word count threshold': 5,
            'Train device': 'cuda',
            'Word2vec kernel size': 2,
            'Word2vec embedding size': 2000,
            'Word2vec train number of epochs': 3,
            'Word2vec train batch size': 2000,
            'Word2vec train learning rate': 1e-3,
            'Classifier hidden layer size': 200,
            'Classifier train number of epochs': 100,
            'Classifier train batch size': 2000,
            'Classifier train learning rate': 0.01,
            'Inference filename': 'custom_political.txt'
            }
        
        with open(params_path, 'w', encoding='utf-8') as file:
            json.dump(my_params, file, indent=2)
    print(my_params)
    main(my_params)
