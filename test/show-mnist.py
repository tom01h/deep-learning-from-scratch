# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

np.set_printoptions(threshold=100)

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False)

sample_image = x_train[1000:1100].reshape((10, 10, 1, 28, 28)).transpose((0, 3, 1, 4, 2)).reshape((280, 280)) # 先頭100個をタイル状に並べ替える
Image.fromarray(np.uint8(sample_image*255)).save('sample.png')
print(t_train[1000:1100].reshape(10,10))
#pil_img = Image.fromarray(np.uint8(sample_image*255))
#pil_img.show()
