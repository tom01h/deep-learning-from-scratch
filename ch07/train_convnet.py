# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import time
import cupy as cp
#import numpy as cp
import numpy as np
import matplotlib.pyplot as plt
from dataset.cifar10 import load_cifar10
from simple_convnet import SimpleConvNet
from common.trainer import Trainer

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_cifar10(normalize=False, flatten=False, one_hot_label=True)

x_train = x_train * 2.0 - 255
x_test = x_test * 2.0 - 255

if os.path.exists("ttarray.pkl"):
    with open("ttarray.pkl", 'rb') as f:
        t_train = pickle.load(f)
        print("Loaded Teacher array!")

# 処理に時間のかかる場合はデータを削減 
#train_mask = np.random.choice(x_train.shape[0], 3000)
#x_train = x_train[train_mask]
#t_train = t_train[train_mask]

max_epochs = 25

network = SimpleConvNet(input_dim=(3,32,32),
                        conv_param = {'filter_num': (32, 32, 64), 'filter_size': 3, 'pad': 1, 'stride': 1},
                        hidden_size=512, output_size=10, weight_init_std=0.01)
                        
trainer = Trainer(network, x_train, t_train, x_test, t_test,
                  epochs=max_epochs, mini_batch_size=100,
                  optimizer='Adam', optimizer_param={'lr': 0.001},
                  evaluate_sample_num_per_epoch=1000, early_stopping=10)
start = time.time()
trainer.train()
elapsed_time = time.time() - start
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

# パラメータの保存
network.save_params("params.pkl")
print("Saved Network Parameters!")

# グラフの描画
markers = {'train': 'o', 'test': 's'}
x = np.arange(trainer.current_epoch)
plt.plot(x, trainer.train_acc_list, marker='o', label='train', markevery=2)
plt.plot(x, trainer.test_acc_list, marker='s', label='test', markevery=2)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.ylim(0, 1.0)
plt.legend(loc='lower right')
plt.show()

np.set_printoptions(threshold=50)
print(np.round(trainer.test_acc_list,3))
