# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import cupy as cp
#import numpy as cp
import numpy as np
from common.optimizer import *

class Trainer:
    """ニューラルネットの訓練を行うクラス
    """
    def __init__(self, network, x_train, t_train, x_test, t_test,
                 epochs=20, mini_batch_size=100,
                 optimizer='SGD', optimizer_param={'lr':0.01}, 
                 evaluate_sample_num_per_epoch=None, early_stopping=5, verbose=True):
        self.network = network
        self.verbose = verbose
        self.x_train = x_train
        self.t_train = t_train
        self.x_test = x_test
        self.t_test = t_test
        self.epochs = epochs
        self.batch_size = mini_batch_size
        self.evaluate_sample_num_per_epoch = evaluate_sample_num_per_epoch

        # optimzer
        optimizer_class_dict = {'sgd':SGD, 'momentum':Momentum, 'nesterov':Nesterov,
                                'adagrad':AdaGrad, 'rmsprpo':RMSprop, 'adam':Adam}
        self.optimizer = optimizer_class_dict[optimizer.lower()](**optimizer_param)
        self.early_stopping = EarlyStopping(patience=early_stopping, verbose=self.verbose)
        
        self.train_size = x_train.shape[0]
        self.iter_per_epoch = max(self.train_size / mini_batch_size, 1)
        self.max_iter = int(epochs * self.iter_per_epoch)
        self.current_iter = 0
        self.current_epoch = 0
        
        self.train_loss_list = []
        self.train_acc_list = []
        self.test_acc_list = []

    def train_step(self):
        early_stopping = False
        batch_mask = np.random.choice(self.train_size, self.batch_size)
        x_batch = cp.array(self.x_train[batch_mask])
        t_batch = cp.array(self.t_train[batch_mask])
        
        grads = self.network.gradient(x_batch, t_batch)
        self.optimizer.update(self.network.params, grads)
        
        loss = self.network.loss(x_batch, t_batch)
        self.train_loss_list.append(loss)
        if self.verbose: print(str(self.current_epoch) + " : " + str(int(self.current_iter % self.iter_per_epoch)) + " : train loss:" + str(loss))
        
        if self.current_iter % self.iter_per_epoch == 0:
            self.current_epoch += 1
            
            if self.evaluate_sample_num_per_epoch is None:
                x_train_sample = cp.array(self.x_train)
                t_train_sample = cp.array(self.t_train)
                x_test_sample = cp.array(self.x_test)
                t_test_sample = cp.array(self.t_test)
            else:
                t = self.evaluate_sample_num_per_epoch
                x_train_sample = cp.array(self.x_train[:t])
                t_train_sample = cp.array(self.t_train[:t])
                x_test_sample = cp.array(self.x_test[:t])
                t_test_sample = cp.array(self.t_test[:t])
                
            train_acc = self.network.accuracy(x_train_sample, t_train_sample)
            test_acc = self.network.accuracy(x_test_sample, t_test_sample)
            self.train_acc_list.append(train_acc)
            self.test_acc_list.append(test_acc)
            early_stopping = self.early_stopping.validate(test_acc)

            if self.verbose: print("=== epoch:" + str(self.current_epoch) + ", train acc:" + str(train_acc) + ", test acc:" + str(test_acc) + " ===")
        self.current_iter += 1
        return early_stopping

    def train(self):
        for i in range(self.max_iter):
            if self.train_step():
                break

        test_acc = self.network.accuracy(cp.array(self.x_test), cp.array(self.t_test))

        if self.verbose:
            print("=============== Final Test Accuracy ===============")
            print("test acc:" + str(test_acc))

