# https://github.com/WegraLee/deep-learning-from-scratch-2
# 여러가지 layers를 딥러닝 라이브러리 없이 구현

# layer class의 구현 규칙:
# 1) 모든 계층은 함수 forward()와 backward()를 가진다
# 2) 모든 계층은 변수 params와 grads를 가진다

import numpy as np
from .function import softmax, cross_entropy_error


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
