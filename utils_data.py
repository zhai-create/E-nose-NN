#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 3/2022
# @Author  : Zhai Shichao
# @FileName: utils_data.py
# Please refer to README.md for details
# 详情请参阅README.md


import scipy.io as scio
import numpy as np
import matplotlib.pyplot as plt
import os

try:  # tf1
    from tensorflow.python.keras.utils import to_categorical
except ImportError:  # tf2
    from tensorflow.keras.utils import to_categorical


def loadmat_1(batch, norm=False, shuffle=True):
    """
    10 boards, use Batch_new1.mat or Batch_new2_norm.mat
    using functools.partial set the default value of norm
    :param shuffle: shuffle or not
    :param norm: directory of .mat file
    :param batch: select 1~10 batch
    :return: [sdata_train, label_train, data_test, label_test] (?,8,16,1) (?, 6)
    """
    assert batch > 0 & batch < 11, "Please input correct batch number"
    if norm is False:
        matdir = './datasets/10boards/Batch_new1.mat'
    else:
        matdir = './datasets/10boards/Batch_new2_norm.mat'
    data = scio.loadmat(matdir)
    label = data['C_label'][0, batch - 1].swapaxes(0, 1)  # (?, 6)
    length = label.shape[0]
    data = data['batch'][0, batch - 1].reshape(length, 16, 8, 1).swapaxes(1, 2)  # (?, 8, 16, 1)
    if shuffle:
        state = np.random.get_state()
        np.random.shuffle(data)
        np.random.set_state(state)
        np.random.shuffle(label)
    data = data.swapaxes(1, 2).swapaxes(0, 1)
    data = data.reshape(data.shape + (1,))
    return data, label


def loadmat_2(batch, shuffle=False):
    """
    4 months, origin .txt
    beer:1, pj
    blacktea:2, hc
    greentea:3, mj
    liquor:4, bj
    olongtea:5, wlc
    puertea:6, pr
    wine:7, ptj
    :param batch:
    :param shuffle:
    :return: selectout (x, n, 180, 1, 1) outlabel (n, 7)
    """
    assert batch == 1 or batch == 2 or batch == 3, "Please input correct batch number"
    if batch == 1:
        skip = 2
    else:
        skip = 0
    labeldict = {'pj': 1, 'hc': 2, 'mj': 3, 'bj': 4, 'wl': 5, 'pr': 6, 'pt': 0,
                 'be': 1, 'bl': 2, 'gr': 3, 'li': 4, 'ol': 5, 'pu': 6, 'wi': 0}
    batch -= 1
    paths = ['./datasets/4months/电子鼻20160518',
             './datasets/4months/电子鼻20160728',
             './datasets/4months/电子鼻20160806']
    outdata = np.ndarray((0, 32, 180, 1, 1))
    outlabel = []
    for file in os.listdir(paths[batch]):
        data = np.loadtxt(paths[batch] + '\\' + file, skiprows=skip)
        data = data.reshape((1, data.shape[0], data.shape[1], 1, 1))
        outdata = np.concatenate((outdata, data))
        outlabel.append(labeldict[os.path.splitext(file)[0][0:2]])
    outlabel = to_categorical(np.array(outlabel), num_classes=7)
    if shuffle:
        state = np.random.get_state()
        np.random.set_state(state)
        np.random.shuffle(outdata)
        np.random.set_state(state)
        np.random.shuffle(outlabel)
        np.random.set_state(state)
    outdata = outdata.swapaxes(0, 1)
    # 去除特定传感器
    selection = [x for x in range(0, 20)] + [23]  # 要保留的传感器ID
    selectout = np.ndarray((0, outdata.shape[1], 180, 1, 1))
    for index in selection:
        selectout = np.concatenate((selectout, outdata[index:index + 1]), axis=0)
    selectout = selectout.swapaxes(0, 2)[130:180][:][:][:][:].swapaxes(0, 2)  # 只取后xx个点
    return selectout, outlabel


def acc_calc(label, result, n=6):
    """
    calculate the final accuracy
    :param label: (None, 6)
    :param result: (None, 6)
    :return: acc[7]: accuracy for 6 classes and overall accuracy
    """
    right = [[0] * n][0]
    wrong = [[0] * n][0]
    acc = []
    result = np.argmax(result, axis=1)  # NMS
    for index in range(result.shape[0]):
        if label[index, result[index]] == 1:
            right[result[index]] = right[result[index]] + 1
        else:
            wrong[result[index]] = wrong[result[index]] + 1  # 有错误，不会影响总精度计算，但会导致各气体精度错误
    for index in range(n):
        if right[index] + wrong[index] != 0:
            acc.append(right[index] / (right[index] + wrong[index]))
        else:
            acc.append(0)
    acc.append(sum(right) / (sum(right) + sum(wrong)))
    return acc


def acc_calc_nms(label, result):
    """
    calculate the final accuracy
    :param label: (None, 6)
    :param result: (None, 1)
    :return: acc: accuracy
    """
    label = nms(label)
    wrong = 0
    for index, l in enumerate(label):
        if l != result[index]:
            wrong += 1
    return (label.shape[0] - wrong) / label.shape[0]


def nms(result):
    """
    NMS
    :param result: (None, 6)
    :return: return calss number [5,2,3,0,1,...]
    """
    return np.argmax(result, axis=1)


def draw(y, title=None, xlabel=None, ylabel=None):
    """
    :param y:
    :param title:
    :param xlabel:
    :param ylabel:
    :return:
    """
    plt.plot(y)
    if title is not None:
        plt.title("{}".format(title))
    if xlabel is not None:
        plt.xlabel("{}".format(xlabel))
    else:
        plt.xlabel("epochs")
    if ylabel is not None:
        plt.ylabel("{}".format(ylabel))
    plt.show()
