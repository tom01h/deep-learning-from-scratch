# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import cupy as cp
#import numpy as cp
import numpy as np
from dataset.cifar10 import load_cifar10
from simple_convnet import SimpleConvNet
from common.functions import *

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_cifar10(normalize=False, flatten=False)

#x_train = x_train[:1000]
#t_train = t_train[:1000]

x_train = x_train * 2.0 - 255
x_test = x_test * 2.0 - 255

batch_size=100

network = SimpleConvNet(input_dim=(3,32,32),
                        conv_param = {'filter_num': (32, 32, 64), 'filter_size': 3, 'pad': 1, 'stride': 1},
                        hidden_size=512, output_size=10, weight_init_std=0.01)

# パラメータの復帰
network.load_params("params.pkl")
print("Loaded Network Parameters!")

tt_array=np.empty((0,10),np.float32)

for i in range(int(x_train.shape[0] / batch_size)):
    tx = x_train[i*batch_size:(i+1)*batch_size]
    tt = network.predict(tx, train_flg=False)
    tt = softmax(tt).get()
    tt_array=np.concatenate((tt_array,tt),axis=0)        

#tt_array=tt_array.reshape(-1,10)
with open("ttarray.pkl", 'wb') as f:
    pickle.dump(tt_array, f)