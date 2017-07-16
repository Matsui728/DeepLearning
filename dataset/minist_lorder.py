# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""
import struct

import urllib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# make cashe for test images
def make_train_images_np(self):
    train_images = 'train-images-idx3-ubyte.gz'

    with open(train_images, 'rb') as f:
        data = f.read()
    header = struct.unpack('>i', data[:4])  # 16バイトヘッダのうちの始め4バイトの並びをヘッダとする．
    num_images = struct.unpack('>i', data[4:8])
    width = struct.unpack('>i', data[8:12])
    height = struct.unpack('>i', data[12:16])
    body = data[16:]  #  実際の画素値は16バイト目から格納されている

    fmt = 'B'* (num_images  * width * height)  # 'B' はunsigned char．10000*28*28個のucharを一気に読み込む
    pixels = struct.unpack(fmt, body)
    # Reshape(10000, 28 , 28)
    images = np.array(pixels).reshape(num_images, width, height)
    np.save('train_images.npy', images)  # ndarrayをファイルに保存する

    return images


# make cashe for test images
def make_test_images_np(self):
    test_images = 't10k-images.idx3-ubyte'

    with open(test_images, 'rb') as f:
        data = f.read()

    header = struct.unpack('>i', data[:4])  # 16バイトヘッダのうちの始め4バイトの並びをヘッダとする．
    num_images = struct.unpack('>i', data[4:8])
    width = struct.unpack('>i', data[8:12])
    height = struct.unpack('>i', data[12:16])
    body = data[16:]  # 実際の画素値は16バイト目から格納されている

    fmt = 'B'* (num_images * width * height)  # 'B' はunsigned char．10000*28*28個のucharを一気に読み込む
    pixels = struct.unpack(fmt, body)
    # Reshape(10000, 28 , 28)
    images = np.array(pixels).reshape(num_images, width, height)
    np.save('test_images.npy', images)  # ndarrayをファイルに保存する

    return images


# make cashe for train labels
def make_train_labeles_np(self):
    train_labels = 'train-labels-idx1-ubyte.gz'

    with open(train_labels, 'rb') as f:
        data = f.read()

    header = struct.unpack('>i', data[:4])  # 8バイトヘッダのうちの始め4バイトの並びをヘッダとする．
    num_items = struct.unpack('>i', data[4:8])
    body = data[8:]

    fmt = 'B'* (num_items)  # 'B' はunsigned char．10000個のucharを一気に読み込む
    items = struct.unpack(fmt, body)
    # Reshape(10000, 28 , 28)
    labels = np.array(items)
    np.save('train_labels.npy', labels)  # ndarrayをファイルに保存する

    return labels


# make cashe for test labels
def make_test_labeles_np(self):
    test_labels = 't10k-labels-idx1-ubyte.gz'
    with open(test_labels, 'rb') as f:
        data = f.read()

    header = struct.unpack('>i', data[:4])  # 8バイトヘッダのうちの始め4バイトの並びをヘッダとする．
    num_items = struct.unpack('>i', data[4:8])
    body = data[8:]

    fmt = 'B'* (num_items)  # 'B' はunsigned char．10000個のucharを一気に読み込む
    items = struct.unpack(fmt, body)
    # Reshape(10000, 28 , 28)
    labels = np.array(items)
    np.save('test_labels.npy', labels)  # ndarrayをファイルに保存する

    return labels


# データが残っていたら使用，残っていなかったらキャッシュ作成
def MnistLoader(ndim=2):
    root_url = 'http://yann.lecun.com/exdb/mnist'
    if not Path('train_images.npy').exists():
        urllib.request.urlretrieve(root_url, 'train-images-idx3-ubyte.gz')  # データファイルのDL
        train_images_data = make_train_images_np()
    else:
        train_images_data = train_images.npy

    if not Path('test_images.npy').exists():
        urllib.request.urlretrieve(root_url, 't10k-images-idx3-ubyte.gz')  # データファイルのDL
        train_images_data = make_test_images_np()
    else:
        test_images_data = test_images.npy

    if not Path('train_labels.npy').exists():
        urllib.request.urlretrieve(root_url, 'train-labels-idx1-ubyte.gz')      # データファイルのDL
        train_labels_data = make_train_labeles_np()
    else:
        train_labels_data = train_labels.npy

    if not Path('test_labels.npy').exists():
        urllib.request.urlretrieve(root_url, 't10k-labels-idx1-ubyte.gz')      # データファイルの
        test_labels_data = make_test_labeles_np()
    else:
        test_labels_data = test_labels.npy

    if ndim == 1:
        train_images_data = train_images_data.reshape(1, 28 * 28)
        test_images_data = test_images_data.reshape(1, 28 * 28)
        train_labels_data = train_labels_data.reshape(1, 28 * 28)
        test_labels_data = test_labels_data.reshape(1, 28 * 28)

    elif ndim == 2:
        train_images_data = train_images_data.reshape(1, 28, 28)
        test_images_data = test_images_data.reshape(1, 28, 28)
        train_labels_data = train_labels_data.reshape(1, 28, 28)
        test_labels_data = test_labels_data.reshape(1, 28, 28)

    elif ndim == 3:
        train_images_data = train_images_data.reshape(-1, 1, 28, 28)
        test_images_data = test_images_data.reshape(-1, 1, 28, 28)
        train_labels_data = train_labels_data.reshape(-1, 1, 28, 28)
        test_labels_data = test_labels_data.reshape(-1, 1, 28, 28)

    return train_images_data, test_images_data, train_labels_data, test_labels_data


if __name__ == '__main__':
    # train_images test_images, train_labels, test_labels = MnistUse()
    # train_images, test_images, train_labels, test_labels = MnistLoader(3)
    train_images, test_images, train_labels, test_labels = MnistLoader(3)
    plt.matshow(train_images[0][0], cmap=plt.cm.gray)
    plt.show()
    plt.matshow(test_images[0][0], cmap=plt.cm.gray)
    plt.show()
    print('train label = {}'. format(train_labels[0]))
    print('test label = {}'. format(test_labels[0]))