# https://github.com/WegraLee/deep-learning-from-scratch-2
# 필요한 Utility functions

from .np import *
import numpy


def preprocess(text):
    # 전처리
    text = text.lower()
    text = text.replace('.', ' .').replace(',', '').replace("'", ' ').replace('?', ' ?')
    words = text.split(' ')

    # 딕셔너리에 담기
    word_to_id = {}
    id_to_word = {}
    for word in words:
        if word not in word_to_id:
            new_id = len(word_to_id)
            word_to_id[word] = new_id
            id_to_word[new_id] = word

    # 단어 ID로 구성된 말뭉치 행렬
    corpus = np.array([word_to_id[w] for w in words])

    return corpus, word_to_id, id_to_word


def create_co_matrix(corpus, window_size=1):
    corpus_size = len(corpus)
    vocab_size = len(set(corpus))
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32)

    # 모든 단어를 조사하면서 주변 단어들에 대한 개수 세기
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
    nx = x / numpy.sqrt(numpy.sum(x**2) + eps)
    ny = y / numpy.sqrt(numpy.sum(y**2) + eps)
    return numpy.dot(nx, ny)


def most_similar(word, word_to_id, id_to_word, co_matrix, top=5):
    if word not in word_to_id:
        print("Could not find word [%s]" % word)
        return

    print('--- Most similar words for [' + word + '] ---')

    word_id = word_to_id[word]
    word_vec = co_matrix[word_id]

    vocab_size = len(id_to_word)
    sim_matrix = numpy.zeros(vocab_size)

    # 모든 vocabulary를 돌면서 similarity 계산
    for i in range(vocab_size):
        sim_matrix[i] = cos_similarity(co_matrix[i], word_vec)

    # top의 개수만큼 높은 similarity를 가지는 단어 출력
    count = 0
    for i in reversed(sim_matrix.argsort()):
        if id_to_word[i] == word:
            continue
        print('[rank %d] %s : %s' % (count+1, id_to_word[i], sim_matrix[i]))
        count += 1
        if count >= top:
            print("\n", end="")
            return


def ppmi(co_matrix, verbose=False, eps=1e-8):
    PPMI_matrix = np.zeros_like(co_matrix, dtype=np.float32)

    # 모든 단어의 발생 빈도수 총합
    N = np.sum(co_matrix)

    # 각 단어의 독립 발생 빈도 행렬
    S = np.sum(co_matrix, axis=0)

    total_count = co_matrix.shape[0] * co_matrix.shape[1]
    count = 0

    for i in range(co_matrix.shape[0]):
        for j in range(co_matrix.shape[1]):
            pmi = np.log2(co_matrix[i, j] * N / (S[i] * S[j]) + eps)
            PPMI_matrix[i, j] = max(0, pmi)

            if verbose:
                count += 1
                if (count+1) % (total_count // 20) == 0:
                    print("%.1f%% Done" % (count / total_count * 100))

    return PPMI_matrix


def visualize_2D(U, word_to_id):
    plt.figure(figsize=(16, 10))
    for word, word_id in word_to_id.items():
        plt.annotate(word, (U[word_id, 0], U[word_id, 1]))
    plt.scatter(U[:, 0], U[:, 1], alpha=0.5)
    plt.show()


def create_contexts_target(corpus, window_size=1):
    # get all target in corpus
    target = corpus[window_size:-window_size]

    contexts = []
    for idx in range(window_size, len(corpus)-window_size):
        context = []
        for word_idx in range(-window_size, window_size+1):
            if word_idx == 0:
                continue
            context.append(corpus[idx+word_idx])
        contexts.append(context)

    return np.array(contexts), np.array(target)


def convert_one_hot(vector, vocab_size):
    N = vector.shape[0]

    if vector.ndim == 1:
        one_hot = np.zeros((N, vocab_size), dtype=np.int32)
        for idx, word_id in enumerate(vector):
            one_hot[idx, word_id] = 1

    elif vector.ndim == 2:
        C = vector.shape[1]
        one_hot = np.zeros((N, C, vocab_size), dtype=np.int32)
        for idx_0, word_ids in enumerate(vector):
            for idx_1, word_id in enumerate(word_ids):
                one_hot[idx_0, idx_1, word_id] = 1

    return one_hot


def clip_grads(grads, max_norm):
    total_norm = 0
    for grad in grads:
        total_norm += np.sum(grad ** 2)
    total_norm = np.sqrt(total_norm)

    rate = max_norm / (total_norm + 1e-6)
    if rate < 1:
        for grad in grads:
            grad *= rate


def to_cpu(x):
    import numpy
    if type(x) == numpy.ndarray:
        return x
    return np.asnumpy(x)


def to_gpu(x):
    import cupy
    if type(x) == cupy.ndarray:
        return x
    return cupy.asarray(x)


def normalize(x):
    if x.ndim == 2:
        s = numpy.sqrt((x * x).sum(1))
        x /= s.reshape((s.shape[0], 1))
    elif x.ndim == 1:
        s = numpy.sqrt((x * x).sum())
        x /= s
    return x


def analogy(a, b, c, word_to_id, id_to_word, word_matrix, top=5, answer=None):
    for word in (a, b, c):
        if word not in word_to_id:
            print('There is no word [%s] in vocab dictionary.' % word)
            return

    print('[analogy] ' + a + ' : ' + b + ' = ' + c + ' : ?')
    a_vec, b_vec, c_vec = word_matrix[word_to_id[a]], word_matrix[word_to_id[b]], word_matrix[word_to_id[c]]
    query_vec = b_vec - a_vec + c_vec
    query_vec = normalize(query_vec)

    similarity = np.dot(word_matrix, query_vec)

    if answer is not None:
        print("==>" + answer + ":" + str(np.dot(word_matrix[word_to_id[answer]], query_vec)))

    count = 0
    for i in (-1 * similarity).argsort():
        if np.isnan(similarity[i]):
            continue
        if id_to_word[i] in (a, b, c):
            continue
        print(' {0}: {1}'.format(id_to_word[i], similarity[i]))

        count += 1
        if count >= top:
            print("\n", end="")
            return
