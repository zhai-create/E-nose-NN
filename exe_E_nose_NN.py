#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 3/2022
# @Author  : Zhai Shichao
# @FileName: exe_E_nose_NN.py
# Please refer to README.md for details
# 详情请参阅README.md


import random

from tensorflow.python.keras import models as km
from utils_data import loadmat_1, acc_calc, loadmat_2, nms, acc_calc_nms
from functools import partial
import scipy.io as scio
import numpy as np
import csv

try:  # tf2
    import tensorflow.compat.v1 as tf
    from tensorflow.compat.v1 import ConfigProto
    from tensorflow.compat.v1 import InteractiveSession
except ImportError:  # tf1
    import tensorflow as tf
from utils_network import ArcFaceTrainable, ArcFaceTrainable7, get_output_function_A, get_output_function_B, network_A, network_B
import utils_MMD as MMD
from utils_MMD import standardization, sigmoid


class E_nose_NN(object):
    def __init__(self, dataset: str):
        assert dataset == 'A' or dataset == 'B'
        self.dataset = dataset
        self.classnum = 6 if self.dataset == 'A' else 7  # 标签个数
        self.sensornum = 16 if self.dataset == 'A' else 21  # 传感器个数
        self.batchnum = 10 if self.dataset == 'A' else 3  # 数据集共有批次数
        self.model = network_A([2, 4, 6], self.sensornum) if dataset == 'A' else network_B([4, 8, 10], self.sensornum)
        self.accrecord_mmdrec = np.ndarray((1, self.batchnum - 1))  # 存放MMD Ensemble精度信息
        self.accrecord_classifier2 = np.ndarray((1, self.batchnum - 1))  # 存放分类器精度信息
        self.accrecord_classifier1 = np.ndarray((1, self.batchnum - 1))  # 存放分类器精度信息
        self.accrecord_classifier3 = np.ndarray((1, self.batchnum - 1))  # 存放分类器精度信息
        self.accrecord_classifier4 = np.ndarray((1, self.batchnum - 1))  # 存放分类器精度信息
        # for MMD-based reclassification
        self._hidden_output = [[0] * self.classnum for i in range(4)]  # 存放隐层输出
        # method
        self.loaddata = partial(loadmat_1, norm=True) if dataset == 'A' else loadmat_2  # 数据导入
        self.get_output = partial(get_output_function_A if dataset == 'A' else get_output_function_B, model=self.model)  # 获取隐层输出
        print("[INFO] Use dataset {}".format(dataset))

    def evaluate(self, setting, iffit):
        """
        运行模型
        @param setting: 选择setting 1或setting 2
        @param iffit: 选择是否训练网络 (注意：新生成模型文件会自动覆盖原有文件)
        @return:
        """
        assert setting == 1 or setting == 2
        assert iffit is True or iffit is False
        if setting == 1:
            self.evaluate_setting1(iffit)
        elif setting == 2:
            self.evaluate_setting2(iffit)

    def evaluate_setting1(self, iffit):
        """
        setting 1
        :return:
        """
        print("[INFO] setting1: iffit: {}".format(iffit))
        name = 'net{}_s1.h5'.format(self.dataset)
        '''Load Data'''
        sdata, slabel = self.loaddata(batch=1, shuffle=True)

        '''Train Network & Save Model'''
        if iffit:
            print("[INFO] Start training")
            history = self._fit(sdata, slabel, epoch=50)
            self.model.save(name)
        else:
            print("[INFO] Load model...")
            if self.dataset == 'A':
                self.model = km.load_model('./h5/' + name, custom_objects={'ArcFaceTrainable': ArcFaceTrainable})
            else:
                self.model = km.load_model('./h5/' + name, custom_objects={'ArcFaceTrainable7': ArcFaceTrainable7})

        '''Predict & Recording acc'''
        for tbatch in range(2, self.batchnum + 1):
            tdata, tlabel = self.loaddata(batch=tbatch, shuffle=True)
            result = self.model.predict([tdata[i] for i in range(self.sensornum)] + [np.ones((tdata.shape[1], self.classnum))])
            self.accrecord_classifier2[0, tbatch - 2] = acc_calc(tlabel, result[0], n=self.classnum)[self.classnum]
            self.accrecord_classifier1[0, tbatch - 2] = acc_calc(tlabel, result[1], n=self.classnum)[self.classnum]
            self.accrecord_classifier3[0, tbatch - 2] = acc_calc(tlabel, result[2], n=self.classnum)[self.classnum]
            self.accrecord_classifier4[0, tbatch - 2] = acc_calc(tlabel, result[3], n=self.classnum)[self.classnum]

            '''MMD-based ensemble'''
            print("[INFO] Batch {}: MMD-based ensemble...".format(tbatch))
            result_final = self._MMD_ensemble([1, 1, 1, 1], sdata, slabel, tdata)
            self.accrecord_mmdrec[0, tbatch - 2] = acc_calc_nms(tlabel, result_final)

            print("[INFO] Batch {}: Validation result:".format(tbatch))
            print("T domain: {}".format(tbatch))
            print("Accuracy: {}, {}, {}, {}, {}\n".format(self.accrecord_mmdrec[0, tbatch - 2],
                                                          self.accrecord_classifier2[0, tbatch - 2],
                                                          self.accrecord_classifier1[0, tbatch - 2],
                                                          self.accrecord_classifier3[0, tbatch - 2],
                                                          self.accrecord_classifier4[0, tbatch - 2]))
        self.acc_print()
        # self.acc_save('net{}_s1'.format(self.dataset))

    def evaluate_setting2(self, iffit):
        """
        setting 2
        :return:
        """
        print("[INFO] setting2: iffit: {}".format(iffit))
        for tbatch in range(2, self.batchnum + 1):
            name = 'net{}_s2_{}.h5'.format(self.dataset, tbatch)
            '''Load Data'''
            sdata, slabel = self.loaddata(batch=tbatch - 1, shuffle=True)
            tdata, tlabel = self.loaddata(batch=tbatch, shuffle=False)

            '''Train Network & Save Model'''
            if iffit:
                print("[INFO] Start training")
                history = self._fit(sdata, slabel, epoch=50)
                self.model.save(name)
            else:
                print("[INFO] Load model...")
                if self.dataset == 'A':
                    self.model = km.load_model('./h5/' + name, custom_objects={'ArcFaceTrainable': ArcFaceTrainable})
                else:
                    self.model = km.load_model('./h5/' + name, custom_objects={'ArcFaceTrainable7': ArcFaceTrainable7})

            '''Predict & Recording acc'''
            result = self.model.predict([tdata[i] for i in range(self.sensornum)] + [np.ones((tdata.shape[1], self.classnum))])
            self.accrecord_classifier2[0, tbatch - 2] = acc_calc(tlabel, result[0], n=self.classnum)[self.classnum]
            self.accrecord_classifier1[0, tbatch - 2] = acc_calc(tlabel, result[1], n=self.classnum)[self.classnum]
            self.accrecord_classifier3[0, tbatch - 2] = acc_calc(tlabel, result[2], n=self.classnum)[self.classnum]
            self.accrecord_classifier4[0, tbatch - 2] = acc_calc(tlabel, result[3], n=self.classnum)[self.classnum]

            '''MMD-based ensemble'''
            print("[INFO] Batch {}: MMD-based ensemble...".format(tbatch))
            result_final = self._MMD_ensemble([1, 1, 1, 1], sdata, slabel, tdata)
            self.accrecord_mmdrec[0, tbatch - 2] = acc_calc_nms(tlabel, result_final)

            print("[INFO] Batch {}: Validation result:".format(tbatch))
            print("T domain: {}".format(tbatch))
            print("Accuracy: {}, {}, {}, {}, {}\n".format(self.accrecord_mmdrec[0, tbatch - 2],
                                                          self.accrecord_classifier2[0, tbatch - 2],
                                                          self.accrecord_classifier1[0, tbatch - 2],
                                                          self.accrecord_classifier3[0, tbatch - 2],
                                                          self.accrecord_classifier4[0, tbatch - 2]))
        self.acc_print()
        # self.acc_save('net{}_s2'.format(self.dataset))

    def save_hiddenlayer(self, h5savingpath='./h5'):
        """
        保存隐层（第二个全连接层）输出至.mat文件，可用于MATLAB绘制PCA图
        @param h5savingpath:
        @return:
        """
        print("[INFO] Save hiddenlayer.")
        hiddenfunc = self.get_output(self.model, 'mainc_dense')
        hidden = []
        '''Load Best Model'''
        print("[INFO] Load model...")
        self.model = km.load_model(h5savingpath + '/net{}_s1.h5'.format(self.dataset),
                                   custom_objects={'ArcFaceTrainable': ArcFaceTrainable})
        for tbatch in range(1, self.batchnum + 1):
            tdata, tlabel = self.loaddata(batch=tbatch, shuffle=False)  # for MATLAB analysing convenience, shuffle should be False if save hidden layer
            print("[INFO] Batch {}, {} samples...".format(tbatch, tdata.shape[1]))
            h = np.ndarray((0, 100))
            for i in range(0, tdata.shape[1]):
                h = np.concatenate((h, hiddenfunc([tdata[:, [i]], np.ones((1, self.classnum))])), axis=0)
            hidden.append(h)
        scio.savemat("./hidden_{}.mat".format(self.dataset), {"batch{}".format(j): hidden[j - 1] for j in range(1, self.batchnum + 1)})

    def _MMD_ensemble(self, bias, sdata, slabel, tdata):
        """
        MMD-based Classifier Ensemble
        与论文中的公式相同，对伪标签打分
        :param sdata: 源域数据
        :param slabel: 源域标签
        :param tdata: 目标域数据
        :return:
        """
        classnum = self.classnum
        sensornum = self.sensornum
        # 获取隐层输出函数
        classifier = [self.get_output(layer_name='mainc_dense'), self.get_output(layer_name='prec1_dense'),
                      self.get_output(layer_name='prec2_dense'), self.get_output(layer_name='prec3_dense')]
        # 按类别提取s样本
        s = [sdata[:, np.where(slabel[:, i] == 1)][:, 0] for i in range(classnum)]
        # 计算并保存隐层输出
        for c in range(len(classifier)):
            for g in range(len(s)):
                if s[g].shape[1] != 0:  # 若当前sdata缺少某种类型气体，则使用之前结果代替
                    self._hidden_output[c][g] = classifier[c]([s[g], np.ones((s[g].shape[0], classnum))])
        # 记录最终结果
        result_final = []
        for i in range(0, tdata.shape[1]):
            result_origin = self.model.predict([tdata[s][[i]] for s in range(sensornum)] + [np.ones((1, classnum))])
            result_nms = list(map(nms, result_origin))
            MMDlog = np.ndarray((len(result_origin), result_origin[0].shape[1]))
            try:
                for c in range(MMDlog.shape[0]):  # c: classifier->4
                    result_c = classifier[c]([tdata[:, [i]], np.ones((1, classnum))])
                    for g in range(MMDlog.shape[1]):  # g: gas->classnum
                        MMDlog[c][g] = MMD.mmd_rbf(self._hidden_output[c][g], result_c)
            except tf.errors.InvalidArgumentError:  # 若数据不足
                result_final.append(result_nms[0][0])
            else:  # 若数据足够
                MMDlog = standardization(np.array(MMDlog))
                MMDlog = list(map(sigmoid, MMDlog))  # sigmoid激活: 使MMD和arcface输出具有相同分布
                credit = np.ndarray((len(result_origin), result_origin[0].shape[1]))  # 4*6
                finalcredit = np.zeros((credit.shape[1], 1))  # 4*1
                for c in range(credit.shape[0]):
                    for g in range(credit.shape[1]):
                        credit[c][g] = (result_origin[c][0][g] / MMDlog[c][g]) * bias[c]  # MMD在分母
                        finalcredit[g] += credit[c][g]
                for g in range(credit.shape[1]):
                    finalcredit[g] /= credit.shape[0]
                minc_index = np.where(finalcredit == max(finalcredit))[0][0]  # 最大score的序号
                result_final.append(minc_index)
        return result_final

    def _fit(self, sdata, slabel, epoch):
        history = self.model.fit([sdata[i] for i in range(self.sensornum)] + [slabel], [slabel, slabel, slabel, slabel],
                                 batch_size=80, epochs=epoch, verbose=2, shuffle=True)
        return history

    def acc_print(self):
        """
        打印精度
        @return:
        """
        print('accrecord_mmdrec: \n', self.accrecord_mmdrec)
        print('accrecord_classifier2: \n', self.accrecord_classifier2)
        print('accrecord_classifier1: \n', self.accrecord_classifier1)
        print('accrecord_classifier3: \n', self.accrecord_classifier3)
        print('accrecord_classifier4: \n', self.accrecord_classifier4)

    def acc_save(self, name):
        """
        保存精度数据到文件
        @param name: 文件名
        @return:
        """
        with open('./accrecord_{}.csv'.format(name), 'w', newline="") as f:
            csv_write = csv.writer(f)
            csv_write.writerow(list(self.accrecord_mmdrec[0]))
            csv_write.writerow(list(self.accrecord_classifier2[0]))
            csv_write.writerow(list(self.accrecord_classifier1[0]))
            csv_write.writerow(list(self.accrecord_classifier3[0]))
            csv_write.writerow(list(self.accrecord_classifier4[0]))


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.set_random_seed(seed)


if __name__ == '__main__':
    if tf.__version__[0] == '2':   # 适配tensorflow 2.x
        tf.disable_v2_behavior()
        config = ConfigProto()
        config.gpu_options.allow_growth = True
        session = InteractiveSession(config=config)

    setup_seed(42)  # 固定随机种子

    E_nose_NN('A').evaluate(setting=1, iffit=False)
    E_nose_NN('A').evaluate(setting=2, iffit=False)
    E_nose_NN('B').evaluate(setting=1, iffit=False)
    E_nose_NN('B').evaluate(setting=2, iffit=False)
