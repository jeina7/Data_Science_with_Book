# https://github.com/WegraLee/deep-learning-from-scratch-2
# 여러가지 layers를 딥러닝 라이브러리 없이 구현

# layer class의 구현 규칙:
# 1) 모든 계층은 함수 forward()와 backward()를 가진다
# 2) 모든 계층은 변수 params와 grads를 가진다

import numpy as np
from .function import softmax, cross_entropy_error
from .util import UnigramSampler


class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []
        self.out = None


    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out


    def backward(self, dout):
        dx = dout * (1 - self.out) * self.out
        return dx


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = [np.zeros_like(W), np.zeros_like(b)]


    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        self.x = x
        return out


    def backward(self, dout):
        W, b = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        db = np.sum(dout, axis=0)

        self.grads[0][...] = dW
        self.grads[1][...] = db
        return dx


class MatMul:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.x = None


    def forward(self, x):
        W, = self.params
        out = np.matmul(x, W)
        self.x = x
        return out


    def backward(self, dout):
        W, = self.params
        dx = np.matmul(dout, W.T)
        dW = np.matmul(self.x.T, dout)
        # deep copy
        self.grads[0][...] = dW
        return dx


class SigmoidWithLoss:
    def __init__(self):
        self.params, self.grads = [], []
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        self.t = t
        self.y = 1 / (1 + np.exp(-x))

        self.loss = cross_entropy_error(np.c_[1 - self.y, self.y], self.t)

        return self.loss

    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = (self.y - sefl.t) * dout / batch_size
        return dx


class SoftmaxWithLoss:
    def __init__(self):
        self.params = []
        self.grads = []
        self.y = None
        self.t = None


    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)

        # one-hot-vector -> label vector
        if self.t.size == self.y.size:
            self.t = self.t.argmax(axis=1)

        loss = cross_entropy_error(self.y, self.t)
        return loss


    def backward(self, dout=1):
        batch_size = self.t.shape[0]

        dx = self.y.copy()
        # true labels만 찾아서 y-t = y-1 (t=1)
        dx[np.arange(batch_size), self.t] -= 1
        dx *= dout
        dx = dx / batch_size
        return dx


class Embedding:
    def __init__(self, W):
        self.params = [W]
        self.grads = [np.zeros_like(W)]
        self.idx = None

    def forward(self, idx):
        W, = self.params
        self.idx = idx
        out = W[idx]
        return out

    def backward(self, dout):
        dW, = self.grads

        # dW의 모든 값을 0으로 덮어씌우기
        dW[...] = 0

        # idx에 해당하는 행에 dout값을 모두 더해주기
        np.add.at(dW, self.idx, dout)
        return None


class EmbeddingDot:
    def __init__(self, W):
        self.embed = Embedding(W)
        self.params = self.embed.params
        self.grads = self.embed.grads
        self.cache = None

    def forward(self, h, idx):
        target_W = self.embed.forward(idx)
        out = np.sum(target_W * h, axis=1)
        self.cache = (h, target_W)
        return out

    def backward(self, dout):
        h, target_W = self.cache
        dout = dout.reshape(dout.shape[0], 1)

        # dot 연산에 대한 backward
        # dout의 값에 각각 서로를 바꿔서 곱해준 값이 gradient값
        dtarget_W = dout * h
        dh = dout * target_W
        # embed 계층 쪽에도 backward 수행
        self.embed.backward(dtarget_W)
        return dh


class NegativeSamplingLoss:
    def __init__(self, W, corpus, power=0.75, sample_size=5):
        self.sample_size = sample_size
        self.sampler = UnigramSampler(corpus, power, sample_size)

        # negative sample size에 정답 sample을 위해 1을 더해줌
        self.loss_layers = [sigmoidWithLoss() for _ in range(sample_size + 1)]
        self.embed_dot_layers = [EmbeddingDot(W) for _ in range(sample_size + 1)]

        self.params, self.grads = [], []
        for layer in self.embed_dot_layers:
            self.params += layer.params
            self.grads += layer.grads

    def forward(self, h, target):
        batch_size = target.shape[0]
        negative_sample = self.sampler.get_negative_sample(target)

        # Positive sample
        score = self.embed_dot_layers[0].forward(h, target)
        correct_label = np.ones(batch_size, dtype=np.int32)
        loss = self.loss_layers[0].forward(score, correct_label)

        # Negative sample
        negative_label = np.zeros(batch_size, dtype=np.int32)
        for i in range(self.sample_size):
            negative_target = negative_sample[:, i]
            score = self.embed_dot_layers[1+i].forward(h, negative_target)
            loss += self.loss_layers[1+i].forward(score, negative_label)

        return loss


    def backward(self, dout=1):
        dh = 0
        for loss_layer, embed_dot_layer in zip(self.loss_layers, self.embed_dot_layers):
            dscore = loss_layer.backward(dout)
            dh += embed_dot_layer.backward(dscore)

        return dh
        
