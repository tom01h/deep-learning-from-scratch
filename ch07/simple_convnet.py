# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import cupy as cp
#import numpy as cp
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class SimpleConvNet:
    """単純なConvNet
    """
    def __init__(self, input_dim=(3, 32, 32),
                 conv_param={'filter_num':(32, 32, 64), 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=512, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2*filter_pad) / filter_stride + 1
        conv_data_size    = int(filter_num[0] *  conv_output_size    *  conv_output_size   )
        pool1_output_size = int(filter_num[1] * (conv_output_size/2) * (conv_output_size/2))
        pool2_output_size = int(filter_num[2] * (conv_output_size/4) * (conv_output_size/4))
        pool3_output_size = int(filter_num[2] * (conv_output_size/8) * (conv_output_size/8))

        # 重みの初期化
        self.params = {}
        self.params['W1'] = cp.array( weight_init_std * \
                            cp.random.randn(filter_num[0], input_dim[0], filter_size, filter_size), dtype=np.float32)

        self.params['W2'] = cp.array( weight_init_std * \
                            cp.random.randn(filter_num[1], filter_num[0], filter_size, filter_size), dtype=np.float32)

        self.params['W3'] = cp.array( weight_init_std * \
                            cp.random.randn(filter_num[2], filter_num[1], filter_size, filter_size), dtype=np.float32)

        self.params['W4'] = cp.array( weight_init_std * \
                            cp.random.randn(pool3_output_size, hidden_size), dtype=np.float32)

        self.params['W5'] = cp.array( weight_init_std * \
                            cp.random.randn(hidden_size, output_size), dtype=np.float32)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['LightNorm1'] = LightNormalization()
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv2'] = BinConvolution(self.params['W2'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['LightNorm2'] = LightNormalization()
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv3'] = BinConvolution(self.params['W3'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['LightNorm3'] = LightNormalization()
        self.layers['Relu3'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine4'] = BinAffine(self.params['W4'])
        self.layers['LightNorm4'] = LightNormalization()
        self.layers['Relu4'] = Relu()

        self.layers['Affine5'] = Affine(self.params['W5'])

        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)

        return x

    def loss(self, x, t):
        """損失関数を求める
        引数のxは入力データ、tは教師ラベル
        """
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t, batch_size=100):
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        
        acc = 0.0
        
        for i in range(int(x.shape[0] / batch_size)):
            tx = x[i*batch_size:(i+1)*batch_size]
            tt = t[i*batch_size:(i+1)*batch_size]
            y = self.predict(tx)
            y = np.argmax(y, axis=1)
            acc += np.sum(y.get() == tt) #cupy
#            acc += np.sum(y == tt) #numpy
        
        return acc / x.shape[0]

    def gradient(self, x, t):
        """勾配を求める（誤差逆伝搬法）

        Parameters
        ----------
        x : 入力データ
        t : 教師ラベル

        Returns
        -------
        各層の勾配を持ったディクショナリ変数
            grads['W1']、grads['W2']、...は各層の重み
            grads['b1']、grads['b2']、...は各層のバイアス
        """
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.last_layer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 設定
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['W2'] = self.layers['Conv2'].dW
        grads['W3'] = self.layers['Conv3'].dW
        grads['W4'] = self.layers['Affine4'].dW
        grads['W5'] = self.layers['Affine5'].dW
        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)
