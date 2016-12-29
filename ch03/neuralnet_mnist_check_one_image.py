# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 親ディレクトリのファイルをインポートするための設定
import numpy as np
import pickle
from dataset.mnist import load_mnist
from common.functions import sigmoid, softmax
from PIL import Image
from pprint import pprint

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network


def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    #pprint_elm('x', x)
    #pprint_elm('W1', W1)
    #pprint_elm('b1', b1)
    a1 = np.dot(x, W1) + b1
    #pprint_elm('a1', a1)
    z1 = sigmoid(a1)
    #pprint_elm('z1', z1)
    #pprint_elm('W2', W2)
    #pprint_elm('b2', b2)
    a2 = np.dot(z1, W2) + b2
    #pprint_elm('a2', a2)
    z2 = sigmoid(a2)
    #pprint_elm('z2', z2)
    #pprint_elm('W3', W3)
    #pprint_elm('b3', b3)
    a3 = np.dot(z2, W3) + b3
    #pprint_elm('a3', a3)
    y = softmax(a3)
    #pprint_elm('y', y)

    return y

def pprint_elm(name,elem):
    pprint('=  '+name+'.shape ========= ' + str(elem.shape))
    pprint('=  '+name+'.ndim ========= ' + str(np.ndim(elem)))
    pprint('= '+name+' =========' + str(elem))
    

def img_show(idx):
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
    img = x_test[idx].reshape(28,28)
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()



x, t = get_data() # MNISTのテスト画像を取得する
network = init_network() # 学習済みの重み、バイアスを取得
index = sys.argv[1] # テスト画像を選ぶ

y = predict(network, x[index]) # ニューラルネットワークと画像を使って判定
p= np.argmax(y) # 最も確率の高い要素のインデックスを取得

img_show(index)
print('predict -> ')
print(p)
print('answer -> ')
print(t[index])
