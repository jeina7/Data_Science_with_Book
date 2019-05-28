# https://github.com/WegraLee/deep-learning-from-scratch-2
# 여러가지 training method들을 딥러닝 라이브러리 없이 구현

import time
import numpy as np
import matplotlib.pyplot as plt
from .util import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.log_step = None


    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, log_step=20, epoch_log_step=False):
        data_size = len(x)
        max_iters = data_size // batch_size
        self.log_step = log_step
        model, optimizer = self.model, self.optimizer
        curr_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # shuffling
            idx = np.random.permutation(np.arange(data_size))
            xs = x[idx]
            ts = t[idx]

            for iters in range(max_iters):
                batch_xs = xs[iters*batch_size:(iters+1)*batch_size]
                batch_ts = ts[iters*batch_size:(iters+1)*batch_size]

                # loss
                loss = model.forward(batch_xs, batch_ts)

                # back propagation
                model.backward()
                # 공유되어서 중복된 가중치는 하나만 남겨두기
                params, grads = remove_duplicate(model.params, model.grads)

                # max_grad값 있으면 그 값에 맞게 조정
                if max_grad is not None:
                    clip_grads(grads, max_grad)

                # update gradient
                optimizer.update(params, grads)

                curr_loss += loss
                loss_count += 1

                # save loss
                if (iters+1) % self.log_step == 0:
                    avg_loss = curr_loss / loss_count
                    self.loss_list.append(avg_loss)
                    curr_loss, loss_count = 0, 0

                # evaluation
                if epoch_log_step:
                    if ((epoch+1) % epoch_log_step == 0) & ((iters+1) % self.log_step == 0):
                        elapsed_time = time.time() - start_time
                        print("time %ds | epoch %d | step %d / %d | loss %.2f" % (elapsed_time, epoch+1, \
                                                                                  iters+1, max_iters, avg_loss))
                else:
                    if (iters+1) % self.log_step == 0:
                        elapsed_time = time.time() - start_time
                        print("time %ds | epoch %d | step %d / %d | loss %.2f" % (elapsed_time, epoch+1, \
                                                                                  iters+1, max_iters, avg_loss))


    def plot(self, ylim=None):
        plt.figure(figsize=(8, 6))
        x = np.arange(len(self.loss_list))
        if ylim:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iteration (x' + str(self.log_step) + ')')
        plt.ylabel('loss')
        plt.show()


def remove_duplicate(params, grads):
    '''
    매개변수 중 중복되는 가중치는 하나로 모아서 가중치 업데이트 시 오류가 없게 한다.
    '''
    while True:
        flag = False
        L = len(params)

        for i in range(0, L-1):
            for j in range(i+1, L):
                if params[i] is params[j]:
                    grads[i] += grads[j]
                    flag = True
                    params.pop(j)
                    grads.pop(j)
                elif (params[i].ndim == 2) and (params[j].ndim == 2) and (params[i].T.shape == params[j].shape) and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    flag = True
                    params.pop(j)
                    grads.pop(j)

                if flag: break
            if flag: break

        if not flag: break

    return params, grads
