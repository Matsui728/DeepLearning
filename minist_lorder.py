# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""
import struct

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
    body = data[16:]  #  実際の画素値は16バイト目から格納されている

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
def make_train_labeles_np(self):
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
def MnistConvert（self):
    if not Path('train_images.npy').exists():
        make_train_images_np()
        train_images_data = train_images.npy
    else:
        train_images_data = train_images.npy

    if not Path('test_images.npy').exists():
        make_test_images_np()
        train_images_data = train_images.npy
    else:
        test_images_data = test_images.npy

    if not Path('train_labels.npy').exists():
        make_train_labels_np()
        train_labels_data = train_labels.npy
    else:
        train_labels_data = trainlabels.npy

    if not Path('test_labels.npy').exists():
        make_test_labels_np()
        test_labels_data = test_labels.npy
    else:
        test_labels_data = test_labels.npy

    return train_images_data, test_images_data, train_labels_data, test_labels_data

if __name__ == '__main__':
