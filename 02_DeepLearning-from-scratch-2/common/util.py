# https://github.com/WegraLee/deep-learning-from-scratch-2
# 필요한 Utility functions

import numpy as np

def preprocess(text):
    text = text.lower()
    text = text.replace('.', ' .').replace(',', '').replace("'", ' ').replace('?', ' ?')
    words = text.split(' ')
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, window_size=1):
    corpus_size = len(corpus)
    vocab_size = len(set(corpus))
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    for idx, word_id in enumerate(corpus):
        for i in range(1, window_size+1):
            left_idx = idx - i
            right_idx = idx + i

            if left_idx >= 0:
                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size:
                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix


def cos_similarity(x, y, eps=1e-8):
    # normalization
    nx = x / np.sqrt(np.sum(x**2) + eps)
    ny = y / np.sqrt(np.sum(y**2) + eps)
    return np.dot(nx, ny)


def most_similar(word, word_to_id, id_to_word, co_matrix, top=5):
    if word not in word_to_id:
        print("Could not find word [%s]" % word)
        return

    print('word: ' + word + '\n')

    word_id = word_to_id[word]
    word_vec = co_matrix[word_id]

    vocab_size = len(id_to_word)
    sim_matrix = np.zeros(vocab_size)

    # 모든 vocabulary를 돌면서 similarity 계산
    for i in range(vocab_size):
        sim_matrix[i] = cos_similarity(co_matrix[i], word_vec)

    # top의 개수만큼 높은 similarity를 가지는 단어 출력
    print('Most Similar words')
    count = 0
    for i in reversed(sim_matrix.argsort()):
        if id_to_word[i] == word:
            continue
        print('[rank %d] %s : %s' % (count+1, id_to_word[i], sim_matrix[i]))
        count += 1
        if count >= top:
            return
