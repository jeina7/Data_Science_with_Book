# https://github.com/WegraLee/deep-learning-from-scratch-2
# 여러가지 dataset 생성기를 딥러닝 라이브러리 없이 구현

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
