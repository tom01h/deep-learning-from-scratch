# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import pickle
import numpy as np
from collections import OrderedDict
from common.layers import *
from common.gradient import numerical_gradient


class SimpleConvNet:
    """単純なConvNet
    """
    def __init__(self, input_dim=(3, 32, 32),
                 conv_param={'filter_num':(32, 32, 32), 'filter_size':3, 'pad':1, 'stride':1},
                 hidden_size=256, output_size=10, weight_init_std=0.01):
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
        self.params['W1'] = weight_init_std * \
                            np.random.randn(filter_num[0], input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num[0])
        self.params['gamma1'] = np.ones(int(conv_data_size))
        self.params['beta1'] = np.zeros(int(conv_data_size))

        self.params['W2'] = weight_init_std * \
                            np.random.randn(filter_num[1], filter_num[0], filter_size, filter_size)
        self.params['b2'] = np.zeros(filter_num[1])
        self.params['gamma2'] = np.ones(int(pool1_output_size))
        self.params['beta2'] = np.zeros(int(pool1_output_size))

        self.params['W3'] = weight_init_std * \
                            np.random.randn(filter_num[2], filter_num[1], filter_size, filter_size)
        self.params['b3'] = np.zeros(filter_num[2])
        self.params['gamma3'] = np.ones(int(pool2_output_size))
        self.params['beta3'] = np.zeros(int(pool2_output_size))

        self.params['W4'] = weight_init_std * \
                            np.random.randn(pool3_output_size, hidden_size)
        self.params['b4'] = np.zeros(hidden_size)
        self.params['gamma4'] = np.ones(hidden_size)
        self.params['beta4'] = np.zeros(hidden_size)

        self.params['W5'] = weight_init_std * \
                            np.random.randn(hidden_size, output_size)
        self.params['b5'] = np.zeros(output_size)

        # レイヤの生成
        self.layers = OrderedDict()
        self.layers['Conv1'] = Convolution(self.params['W1'], self.params['b1'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['BatchNorm1'] = BatchNormalization(self.params['gamma1'], self.params['beta1'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv2'] = Convolution(self.params['W2'], self.params['b2'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['BatchNorm2'] = BatchNormalization(self.params['gamma2'], self.params['beta2'])
        self.layers['Relu2'] = Relu()
        self.layers['Pool2'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Conv3'] = Convolution(self.params['W3'], self.params['b3'],
                                           conv_param['stride'], conv_param['pad'])
        self.layers['BatchNorm3'] = BatchNormalization(self.params['gamma3'], self.params['beta3'])
        self.layers['Relu3'] = Relu()
        self.layers['Pool3'] = Pooling(pool_h=2, pool_w=2, stride=2)

        self.layers['Affine4'] = Affine(self.params['W4'], self.params['b4'])
        self.layers['BatchNorm4'] = BatchNormalization(self.params['gamma4'], self.params['beta4'])
        self.layers['Relu4'] = Relu()

        self.layers['Affine5'] = Affine(self.params['W5'], self.params['b5'])

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
            acc += np.sum(y == tt) 
        
        return acc / x.shape[0]

    def numerical_gradient(self, x, t):
        """勾配を求める（数値微分）

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
        loss_w = lambda w: self.loss(x, t)

        grads = {}
        for idx in (1, 2, 3):
            grads['W' + str(idx)] = numerical_gradient(loss_w, self.params['W' + str(idx)])
            grads['b' + str(idx)] = numerical_gradient(loss_w, self.params['b' + str(idx)])

        return grads

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
        grads['W1'], grads['b1'] = self.layers['Conv1'].dW, self.layers['Conv1'].db
        grads['W2'], grads['b2'] = self.layers['Conv2'].dW, self.layers['Conv2'].db
        grads['W3'], grads['b3'] = self.layers['Conv3'].dW, self.layers['Conv3'].db
        grads['W4'], grads['b4'] = self.layers['Affine4'].dW, self.layers['Affine4'].db
        grads['W5'], grads['b5'] = self.layers['Affine5'].dW, self.layers['Affine5'].db
        grads['gamma1'] =  self.layers['BatchNorm1'].dgamma
        grads['beta1'] = self.layers['BatchNorm1'].dbeta
        grads['gamma2'] =  self.layers['BatchNorm2'].dgamma
        grads['beta2'] = self.layers['BatchNorm2'].dbeta
        grads['gamma3'] =  self.layers['BatchNorm3'].dgamma
        grads['beta3'] = self.layers['BatchNorm3'].dbeta
        grads['gamma4'] =  self.layers['BatchNorm4'].dgamma
        grads['beta4'] = self.layers['BatchNorm4'].dbeta
        return grads
        
    def save_params(self, file_name="params.pkl"):
        params = {}
        for key, val in self.params.items():
            params[key] = val
        with open(file_name, 'wb') as f:
            pickle.dump(params, f)

    def load_params(self, file_name="params.pkl"):
        with open(file_name, 'rb') as f:
            params = pickle.load(f)
        for key, val in params.items():
            self.params[key] = val

        for i, key in enumerate(['Conv1', 'Affine1', 'Affine2']):
            self.layers[key].W = self.params['W' + str(i+1)]
            self.layers[key].b = self.params['b' + str(i+1)]