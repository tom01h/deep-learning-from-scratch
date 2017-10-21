# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import time
import cupy as cp
#import numpy as cp
import numpy as np
from dataset.cifar10 import load_cifar10
from simple_convnet import SimpleConvNet

# データの読み込み
(x_train, t_train), (x_test, t_test) = load_cifar10(normalize=False, flatten=False, one_hot_label=True)

x_train = x_train * 2.0 - 255
x_test = x_test * 2.0 - 255

# 処理に時間のかかる場合はデータを削減 
#train_mask = np.random.choice(x_train.shape[0], 5000)
#x_train = x_train[train_mask]
#t_train = t_train[train_mask]
#test_mask = np.random.choice(x_test.shape[0], 1000)
x_test = x_test[:1000]
t_test = t_test[:1000]

t_test =cp.array(t_test, np.float32)

network = SimpleConvNet(input_dim=(3,32,32),
                        conv_param = {'filter_num': (32, 32, 64), 'filter_size': 3, 'pad': 1, 'stride': 1},
                        hidden_size=512, output_size=10, weight_init_std=0.01)

# パラメータの復帰
network.load_params("params.pkl")
print("Loaded Network Parameters!")

start = time.time()
test_acc = network.accuracy(x_test, t_test)
elapsed_time = time.time() - start
print ("=== " + "test acc:" + str(test_acc) + " ===")
print ("elapsed_time:{0}".format(elapsed_time) + "[sec]")

#network.save_params("params.pkl")
#print("Saved Network Parameters!")

np.savetxt('W1.h', network.params['W1'].reshape(32,-1), delimiter=',', newline=',\n', header='float W1[32][27]={', footer='};', comments='')
np.savetxt('mean1.h', network.layers['LightNorm1'].running_mean.get(), newline=',\n', header='float mean1[32]={', footer='};', comments='')
np.savetxt('var1.h', network.layers['LightNorm1'].running_var.get(), newline=',\n', header='float var1[32]={', footer='};', comments='')
np.savetxt('W2.h', network.layers['Conv2'].col_W.get().T.astype(np.int), fmt='%d', delimiter=',', newline=',\n', header='int W2[32][32*9]={', footer='};', comments='')
np.savetxt('mean2.h', network.layers['LightNorm2'].running_mean.get(), newline=',\n', header='float mean2[32]={', footer='};', comments='')
np.savetxt('var2.h', network.layers['LightNorm2'].running_var.get(), newline=',\n', header='float var2[32]={', footer='};', comments='')
np.savetxt('W3.h', network.layers['Conv3'].col_W.get().T.astype(np.int), fmt='%d', delimiter=',', newline=',\n', header='int W3[64][32*9]={', footer='};', comments='')
np.savetxt('mean3.h', network.layers['LightNorm3'].running_mean.get(), newline=',\n', header='float mean3[64]={', footer='};', comments='')
np.savetxt('var3.h', network.layers['LightNorm3'].running_var.get(), newline=',\n', header='float var3[64]={', footer='};', comments='')
np.savetxt('W4.h', network.layers['Affine4'].bW.get(), fmt='%d', delimiter=',', newline=',\n', header='int W4[512][1024]={', footer='};', comments='')
np.savetxt('mean4.h', network.layers['LightNorm4'].running_mean.get(), newline=',\n', header='float mean4[512]={', footer='};', comments='')
np.savetxt('var4.h', network.layers['LightNorm4'].running_var.get(), newline=',\n', header='float var4[512]={', footer='};', comments='')
np.savetxt('W5.h', network.params['W5'].T, delimiter=',', newline=',\n', header='float W5[10][512]={', footer='};', comments='')

#np.set_printoptions(threshold=50)
