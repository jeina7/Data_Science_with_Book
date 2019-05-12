# 여러가지 종류의 Net을 딥러닝 라이브러리 없이 구현

# Net class의 구현 규칙:
# 1) 모든 Net은 함수 predict()를 가진다
# 2) 모든 Net은 그 안에 포함된 layers의 params를 모두 합친 Net의 새로운 변수 params를 가진다

import numpy as np
from layer import Sigmoid, Affine


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size):
        self.input_size, self.hidden_size, self.output_size = input_size, hidden_size, output_size

        # weight, bias initialize
        W1 = np.random.randn(self.input_size, self.hidden_size)
        b1 = np.random.randn(self.hidden_size)
        W2 = np.random.randn(self.hidden_size, self.output_size)
        b2 = np.random.randn(self.output_size)

        # layers
        self.layers = [
            Affine(W1, b1),
            Sigmoid(),
            Affine(W2, b2)
        ]

        # collect params
        self.params = []
        for layer in self.layers:
            self.params += layer.params

    def predict(self, x):
        for layer in self.layers:
            x = layer.forward(x)
        return x
