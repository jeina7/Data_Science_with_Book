# https://github.com/WegraLee/deep-learning-from-scratch-2
# 여러가지 optimizer를 딥러닝 라이브러리 없이 구현

import numpy as np


class SGD:
    def __init__(self, lr=0.01):
        self.lr = lr


    def update(self, params, grads):
        for i in range(len(params)):
            params[i] -= self.lr * grads[i]
