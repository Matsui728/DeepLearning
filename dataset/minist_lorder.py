# -*- coding: utf-8 -*-
"""
Spyderエディタ

これは一時的なスクリプトファイルです
"""

import urllib
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

import gzip


# make cashe for train images
def make_train_images_np():
    train_images = 'train-images-idx3-ubyte.gz'

    with gzip.open(train_images, 'rb') as f:
        data = f.read()

    num_images = int.from_bytes(data[4:8], 'big')
    width = int.from_bytes(data[8:12], 'big')
    height = int.from_bytes(data[12:16], 'big')
    pixels = np.frombuffer(data, np.uint8, -1, 16)

    # Reshape(60000, 28 , 28, 1)
    images = pixels.reshape(num_images, width, height, 1)
    np.save('train_images.npy', images)  # ndarrayをファイルに保存する

    print('Making cashe for train images is completed.')
    return images


# make cashe for test images
def make_test_images_np():
    test_images = 't10k-images-idx3-ubyte.gz'

    with gzip.open(test_images, 'rb') as f:
        data = f.read()

    num_images = int.from_bytes(data[4:8], 'big')
    width = int.from_bytes(data[8:12], 'big')
    height = int.from_bytes(data[12:16], 'big')
    pixels = np.frombuffer(data, np.uint8, -1, 16)

    # Reshape(10000, 28 , 28, 1)
    images = pixels.reshape(num_images, width, height, 1)
    np.save('test_images.npy', images)  # ndarrayをファイルに保存する

    print('Making cashe for test images is completed.')
    return images


# make cashe for train labels
def make_train_labeles_np():
    train_labels = 'train-labels-idx1-ubyte.gz'

    with gzip.open(train_labels, 'rb') as f:
        data = f.read()

    labels = np.frombuffer(data, np.uint8, -1, 8)
    np.save('train_labels.npy', labels)  # ndarrayをファイルに保存する

    print('Making cashe for train labels is completed.')
    return labels


# make cashe for test labels
def make_test_labeles_np():
    test_labels = 't10k-labels-idx1-ubyte.gz'
    with gzip.open(test_labels, 'rb') as f:
        data = f.read()

    labels = np.frombuffer(data, np.uint8, -1, 8)
    np.save('test_labels.npy', labels)  # ndarrayをファイルに保存する

    print('Making cashe for test labels is completed.')
    return labels


# データが残っていたら使用，残っていなかったらキャッシュ作成
def MnistLoader(ndim=2):
    root_url = 'http://yann.lecun.com/exdb/mnist'
    if not Path('train_images.npy').exists():
        urllib.request.urlretrieve(root_url + '/train-images-idx3-ubyte.gz',
                                   'train-images-idx3-ubyte.gz')  # データファイルのDL
        train_images_data = make_train_images_np()
    else:
        train_images_data = train_images.npy

    if not Path('test_images.npy').exists():
        urllib.request.urlretrieve(root_url + '/t10k-images-idx3-ubyte.gz',
                                   't10k-images-idx3-ubyte.gz')  # データファイルのDL
        test_images_data = make_test_images_np()
    else:
        test_images_data = test_images.npy

    if not Path('train_labels.npy').exists():
        urllib.request.urlretrieve(root_url + '/train-labels-idx1-ubyte.gz',
                                   'train-labels-idx1-ubyte.gz')   # データファイルのDL
        train_labels_data = make_train_labeles_np()
    else:
        train_labels_data = train_labels.npy

    if not Path('test_labels.npy').exists():
        urllib.request.urlretrieve(root_url + '/t10k-labels-idx1-ubyte.gz',
                                   't10k-labels-idx1-ubyte.gz')   # データファイルのDL
        test_labels_data = make_test_labeles_np()
    else:
        test_labels_data = test_labels.npy

    if ndim == 1:
        train_images_data = train_images_data.reshape(1, 28 * 28)
        test_images_data = test_images_data.reshape(1, 28 * 28)

    elif ndim == 2:
        train_images_data = train_images_data.reshape(1, 28, 28)
        test_images_data = test_images_data.reshape(1, 28, 28)

    elif ndim == 3:
        train_images_data = train_images_data.reshape(-1, 1, 28, 28)
        test_images_data = test_images_data.reshape(-1, 1, 28, 28)

    else:
        raise ValueError('You need define ndim between from 1 to 3.')

    return train_images_data, test_images_data, train_labels_data, test_labels_data


if __name__ == '__main__':
    train_images, test_images, train_labels, test_labels = MnistLoader(3)
    plt.matshow(train_images[0][0], cmap=plt.cm.gray)
    plt.show()
    plt.matshow(test_images[0][0], cmap=plt.cm.gray)
    plt.show()
    print('train label = {}'. format(train_labels[0]))
    print('test label = {}'. format(test_labels[0]))
