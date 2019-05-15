# https://github.com/WegraLee/deep-learning-from-scratch-2
# 여러가지 training method들을 딥러닝 라이브러리 없이 구현

import time
import numpy as np
import matplotlib.pyplot as plt


class Trainer:
    def __init__(self, model, optimizer):
        self.model = model
        self.optimizer = optimizer
        self.loss_list = []
        self.log_step = None


    def fit(self, x, t, max_epoch=10, batch_size=32, max_grad=None, log_step=20):
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

                # update gradient
                optimizer.update(model.params, model.grads)

                curr_loss += loss
                loss_count += 1

                # evaluation
                if (iters+1) % self.log_step == 0:
                    avg_loss = curr_loss / loss_count
                    self.loss_list.append(avg_loss)
                    curr_loss, loss_count = 0, 0
                    elapsed_time = time.time() - start_time
                    print("time %ds | epoch %d | step %d / %d | loss %.2f" % (elapsed_time, epoch+1, iters+1, max_iters, avg_loss))


    def plot(self, ylim=None):
        plt.figure(figsize=(8, 6))
        x = np.arange(len(self.loss_list))
        if ylim:
            plt.ylim(*ylim)
        plt.plot(x, self.loss_list, label='train')
        plt.xlabel('iteration (x' + str(self.log_step) + ')')
        plt.ylabel('loss')
        plt.show()
