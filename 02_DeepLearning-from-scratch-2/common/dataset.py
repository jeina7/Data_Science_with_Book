# https://github.com/WegraLee/deep-learning-from-scratch-2
# 여러가지 dataset 생성기를 딥러닝 라이브러리 없이 구현

import sys
import os
sys.path.append('..')
import urllib.request
import pickle
import numpy as np


class Spiral:
    def __init__(self, sample_num=100, feature_num=2, class_num=3):
        self.sample_num = sample_num
        self.feature_num = feature_num
        self.class_num = class_num


    def load_data(self, seed=2019):
        np.random.seed(seed)

        x = np.zeros((self.sample_num * self.class_num, self.feature_num))
        t = np.zeros((self.sample_num * self.class_num, self.class_num), dtype=np.int)

        for j in range(self.class_num):
            for i in range(self.sample_num):
                rate = i / self.sample_num
                radius = 1 * rate
                theta = j * 4 + 4 * rate + np.random.randn() * 0.2

                idx = self.sample_num * j + i
                x[idx] = np.array([radius * np.sin(theta), radius * np.cos(theta)]).flatten()

                t[idx, j] = 1
        return x, t


class PTB:

    def __init__(self):
        self.url_base = 'https://raw.githubusercontent.com/tomsercu/lstm/master/data/'
        self.key_file = {
            'train':'ptb.train.txt',
            'test':'ptb.test.txt',
            'valid':'ptb.valid.txt'
        }
        self.save_file = {
            'train':'ptb.train.npy',
            'test':'ptb.test.npy',
            'valid':'ptb.valid.npy'
        }
        self.vocab_file = 'ptb.vocab.pkl'
        self.dataset_dir = os.path.dirname(os.path.abspath("__file__")) + '/' + 'ptb_dataset'
        self.file_name = ""


    def _download(self, file_name):
        self.file_name = self.file_name
        file_path = self.dataset_dir + '/' + self.file_name
        if os.path.exists(file_path):
            return

        if not os.path.exists(self.dataset_dir):
            os.mkdir(self.dataset_dir)

        print('Downloading ' + self.file_name + ' ... ')

        try:
            urllib.request.urlretrieve(self.url_base + self.file_name, file_path)
        except urllib.error.URLError:
            import ssl
            ssl._create_default_https_context = ssl._create_unverified_context
            urllib.request.urlretrieve(self.url_base + self.file_name, file_path)

        print('Done')


    def load_vocab(self):
        vocab_path = self.dataset_dir + '/' + self.vocab_file

        if os.path.exists(vocab_path):
            with open(vocab_path, 'rb') as f:
                word_to_id, id_to_word = pickle.load(f)
            return word_to_id, id_to_word

        word_to_id = {}
        id_to_word = {}
        data_type = 'train'
        self.file_name = self.key_file[data_type]
        file_path = self.dataset_dir + '/' + self.file_name

        self._download(self.file_name)

        words = open(file_path).read().replace('\n', '<eos>').strip().split()

        for i, word in enumerate(words):
            if word not in word_to_id:
                tmp_id = len(word_to_id)
                word_to_id[word] = tmp_id
                id_to_word[tmp_id] = word

        with open(vocab_path, 'wb') as f:
            pickle.dump((word_to_id, id_to_word), f)

        return word_to_id, id_to_word


    def load_data(self, data_type='train'):
        '''
            :param data_type: 데이터 유형: 'train' or 'test' or 'valid (val)'
            :return:
        '''
        if data_type == 'val': data_type = 'valid'
        save_path = self.dataset_dir + '/' + self.save_file[data_type]

        word_to_id, id_to_word = self.load_vocab()

        if os.path.exists(save_path):
            corpus = np.load(save_path)
            return corpus, word_to_id, id_to_word

        self.file_name = self.key_file[data_type]
        file_path = self.dataset_dir + '/' + self.file_name
        self._download(self.file_name)

        words = open(file_path).read().replace('\n', '<eos>').strip().split()
        corpus = np.array([word_to_id[w] for w in words])

        np.save(save_path, corpus)
        return corpus, word_to_id, id_to_word
