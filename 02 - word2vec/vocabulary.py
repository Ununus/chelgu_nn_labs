import doc_processor as dproc

class Vocabulary:
    '''Словарь из слов набора документов'''
    def __init__(self, docs_dataset : dproc.Dataset, word_count_threshold : int = 0) -> None:
        self.word2ix_ = dict()
        self.ix2word_ = list()
        if len(docs_dataset.get_train()) == 0:
            raise 'Создание словаря над пустым набором документов'
        if word_count_threshold > 0:
            print('Слово должно встретится хотя бы {} раз(а), чтобы попасть в словарь'.format(word_count_threshold + 1))
        print('Создание словаря из набора документов...')
        word2cnt = dict()
        for (tokenized_text, _) in docs_dataset.get_train():
            for word in tokenized_text:
                word2cnt[word] = word2cnt.get(word, 0) + 1
        for word, cnt in word2cnt.items():
            if cnt > word_count_threshold and word not in self.word2ix_:
                self.word2ix_[word] = len(self.ix2word_)
                self.ix2word_.append(word)
        if word_count_threshold > 0:
            print('Кол-во уникальных слов в документах: {}'.format(len(word2cnt)))
            print('Убрано слов по порогу: {}'.format(len(word2cnt) - len(self.word2ix_)))
        print('Итоговый размер словаря: {}'.format(len(self.word2ix_)))

    def word_to_index(self, word : str):
        return self.word2ix_[word]
    
    def index_to_word(self, idx : int):
        return self.ix2word_[idx]
    
    def has_word(self, word : str):
        return word in self.word2ix_

    def __len__(self):
        return len(self.ix2word_)