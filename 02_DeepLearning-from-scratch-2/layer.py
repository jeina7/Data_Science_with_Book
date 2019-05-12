# 여러가지 layers를 딥러닝 라이브러리 없이 구현

# layer class의 구현 규칙:
# 1) 모든 계층은 함수 forward()와 backward()를 가진다
# 2) 모든 계층은 변수 params와 grads를 가진다

import numpy as np


class Sigmoid:
    def __init__(self):
        self.params = []
        self.grads = []

    def forward(self, x):
        return  1 / (1 + np.exp(-x))

    def backward(self):
        return


class Affine:
    def __init__(self, W, b):
        self.params = [W, b]
        self.grads = []

    def forward(self, x):
        W, b = self.params
        out = np.matmul(x, W) + b
        return out

    def backward(self):
        return
