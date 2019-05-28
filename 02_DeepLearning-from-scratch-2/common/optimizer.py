# https://github.com/WegraLee/deep-learning-from-scratch-2
# 여러가지 optimizer를 딥러닝 라이브러리 없이 구현

import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr


    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]


class Adam:
    '''
    Adam (http://arxiv.org/abs/1412.6980v8)
    '''
    def __init__(self, lr=0.001, beta_1=0.9, beta_2=0.999):
        self.lr = lr
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.iter = 0
        self.m = None
        self.v = None

    def update(self, params, grads):
        if self.m is None:
            self.m, self.v = [], []
            for param in params:
                self.m.append(np.zeros_like(param))
                self.v.append(np.zeros_like(param))

        # learning rate update
        self.iter += 1
        lr_t = self.lr * np.sqrt(1.0 - self.beta_2**self.iter) / (1.0 - self.beta_1**self.iter)

        # params update
        for i in range(len(params)):
            self.m[i] += (1 - self.beta_1) * (grads[i] - self.m[i])
            self.v[i] += (1 - self.beta_2) * (grads[i]**2 - self.v[i])

            params[i] -= lr_t * self.m[i] / (np.sqrt(self.v[i]) + 1e-7)
