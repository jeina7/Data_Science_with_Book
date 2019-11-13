# https://github.com/WegraLee/deep-learning-from-scratch-2
# 여러가지 training method들을 딥러닝 라이브러리 없이 구현

import time
import numpy
import matplotlib.pyplot as plt
from .np import *
from .util import clip_grads


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []


    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, loss_save_step=20, log_step=False):
        data_size = len(x)
        max_iters = data_size // batch_size
        model, optimizer = self.model, self.optimizer
        self.loss_save_step = loss_save_step
        curr_loss = 0
        loss_count = 0

        start_time = time.time()
        for epoch in range(max_epoch):
            # shuffling
            idx = numpy.random.permutation(numpy.arange(data_size))
            x = x[idx]
            t = t[idx]

            for iters in range(max_iters):
                batch_x = x[iters*batch_size:(iters+1)*batch_size]
                batch_t = t[iters*batch_size:(iters+1)*batch_size]

                # loss
                loss = model.forward(batch_x, batch_t)

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
                if (iters+1) % self.loss_save_step == 0:
                    avg_loss = curr_loss / loss_count
                    self.loss_list.append(avg_loss)
                    curr_loss, loss_count = 0, 0

                # log losses
                if (iters+1) % log_step == 0:
                    elapsed_time = time.time() - start_time
                    print("time %ds | epoch %d | step %d / %d | loss %.2f" % (elapsed_time, epoch+1, \
                                                                              iters+1, max_iters, avg_loss))


    def plot(self, ylim=None):
        plt.figure(figsize=(8, 6))
        x = numpy.arange(len(self.loss_list))
        if ylim:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iteration (x' + str(self.loss_save_step) + ')')
        plt.ylabel('loss')
        plt.show()


class RnnlmTrainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.time_idx = None
        self.ppl_list = None
        self.log_step = None


    def get_batch(self, xs, ts, batch_size, time_size):
        batch_x = np.empty((batch_size, time_size), dtype='i')
        batch_t = np.empty((batch_size, time_size), dtype='i')

        data_size = len(xs)
        jump = data_size // batch_size
        offsets = [i * jump for i in range(batch_size)]

        for t in range(time_size):
            for i, offset in enumerate(offsets):
                batch_x[i, t] = xs[(offset + self.time_idx) % data_size]
                batch_t[i, t] = ts[(offset + self.time_idx) % data_size]
            self.time_idx += 1

        return batch_x, batch_t


    def fit(self, xs, ts, max_epoch=10, batch_size=20, time_size=35, max_grad=None, \
            loss_save_step=20, log_step=20):
        data_size = len(xs)
        max_iters = data_size // (batch_size * time_size)
        self.time_idx = 0
        self.ppl_list = []
        self.loss_save_step = loss_save_step
        model, optimizer = self.model, self.optimizer
        total_loss, loss_count = 0, 0

        start_time = time.time()
        for epoch in range(max_epoch):
            for iters in range(max_iters):
                batch_x, batch_t = self.get_batch(xs, ts, batch_size, time_size)

                # loss
                loss = model.forward(batch_x, batch_t)

                # back propagation
                model.backward()
                # 공유되어서 중복된 가중치는 하나만 남겨두기
                params, grads = remove_duplicate(model.params, model.grads)

                # max_grad값 있으면 그 값에 맞게 조정
                if max_grad is not None:
                    clip_grads(grads, max_grad)

                # update gradient
                optimizer.update(params, grads)

                total_loss += loss
                loss_count += 1

                # save loss
                if (iters+1) % self.loss_save_step == 0:
                    avg_loss = total_loss / loss_count
                    ppl = np.exp(avg_loss)
                    self.ppl_list.append(float(ppl))
                    total_loss, loss_count = 0, 0

                # log losses
                if (iters+1) % log_step == 0:
                    elapsed_time = time.time() - start_time
                    print("time %ds | epoch %d | step %d / %d | ppl %.2f" % (elapsed_time, epoch+1, \
                                                                             iters+1, max_iters, ppl))


    def plot(self, ylim=None):
        x = numpy.arange(len(self.ppl_list))
        if ylim is not None:
            plt.ylim(*ylim)
        plt.plot(x, self.ppl_list, label='train')
        plt.xlabel('iter (x' + str(self.loss_save_step) + ')')
        plt.ylabel('perplexity')
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
                elif (params[i].ndim == 2) and (params[j].ndim == 2) and \
                     (params[i].T.shape == params[j].shape) and np.all(params[i].T == params[j]):
                    grads[i] += grads[j].T
                    flag = True
                    params.pop(j)
                    grads.pop(j)

                if flag: break
            if flag: break

        if not flag: break

    return params, grads
