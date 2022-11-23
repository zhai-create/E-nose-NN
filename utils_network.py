#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# @Time    : 3/2022
# @Author  : Zhai Shichao
# @FileName: utils_network.py
# Please refer to README.md for details
# 详情请参阅README.md


from tensorflow.python.keras import layers as kl
from tensorflow.python.keras import models as km
from tensorflow.python.keras import backend as K
from tensorflow.python.keras.layers import Layer
from tensorflow.python.keras import regularizers
import tensorflow as tf


class ArcFaceTrainable(Layer):
    def __init__(self, n_classes=6, regularizer=None, **kwargs):
        super(ArcFaceTrainable, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFaceTrainable, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1].value, self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)
        self.s = 4
        self.m = 0.5

    def call(self, inputs):
        x, y = inputs
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return None, self.n_classes


class ArcFaceTrainable7(Layer):
    def __init__(self, n_classes=7, regularizer=None, **kwargs):
        super(ArcFaceTrainable7, self).__init__(**kwargs)
        self.n_classes = n_classes
        self.regularizer = regularizers.get(regularizer)

    def build(self, input_shape):
        super(ArcFaceTrainable7, self).build(input_shape[0])
        self.W = self.add_weight(name='W',
                                 shape=(input_shape[0][-1].value, self.n_classes),
                                 initializer='glorot_uniform',
                                 trainable=True,
                                 regularizer=self.regularizer)
        self.s = 4
        self.m = 0.5

    def call(self, inputs):
        x, y = inputs
        x = tf.nn.l2_normalize(x, axis=1)
        W = tf.nn.l2_normalize(self.W, axis=0)
        logits = x @ W
        theta = tf.acos(K.clip(logits, -1.0 + K.epsilon(), 1.0 - K.epsilon()))
        target_logits = tf.cos(theta + self.m)
        logits = logits * (1 - y) + target_logits * y
        logits *= self.s
        out = tf.nn.softmax(logits)
        return out

    def compute_output_shape(self, input_shape):
        return None, self.n_classes


def network_A(param, sensornum, summary=False):
    def branch_create(scope):
        assert type(scope) == str, 'TypeError: scope should be a str'
        inputs = kl.Input(shape=(8, 1, 1), name=scope + '/Input')
        # BN = kl.BatchNormalization(1)(inputs)  # !!! batch5和batch10需要去掉BN层才能正常工作，原因不明，猜测需要将8*16的数据一起BN，而不能分传感器BN(8*1)
        conv1 = kl.Conv2D(filters=param[0], kernel_size=(2, 1), padding='same', strides=(1, 1), activation='elu')(inputs)
        conv2 = kl.Conv2D(filters=param[1], kernel_size=(2, 1), padding='same', strides=(1, 1), activation='elu')(conv1)
        conv3 = kl.Conv2D(filters=param[2], kernel_size=(2, 1), padding='same', strides=(1, 1), activation='elu')(conv2)
        return inputs, conv2, conv1, conv3

    # label (for Arcface)
    label = kl.Input(shape=(6,), name='Arcface_input')

    # create branch
    branch = {}
    for branch_num in range(1, sensornum + 1):
        branch_name = 'branch{}'.format(branch_num)
        branch[branch_name] = branch_create(branch_name)

    # preclassifier 1
    preclassifier1 = kl.Concatenate(axis=1, name='preclassfier1')(
        [branch['branch1'][2], branch['branch2'][2], branch['branch3'][2],
         branch['branch4'][2], branch['branch5'][2], branch['branch6'][2],
         branch['branch7'][2], branch['branch8'][2], branch['branch9'][2],
         branch['branch10'][2], branch['branch11'][2], branch['branch12'][2],
         branch['branch13'][2], branch['branch14'][2], branch['branch15'][2],
         branch['branch16'][2]])
    preclassifier1 = kl.Flatten()(preclassifier1)
    preclassifier1 = kl.Dense(units=100, activation='relu')(preclassifier1)
    preclassifier1 = kl.Dense(units=100, activation='relu', name='prec1_dense')(preclassifier1)
    preclassifier1 = ArcFaceTrainable(n_classes=6, name='prec1')([preclassifier1, label])

    # preclassifier 2
    preclassifier2 = kl.Concatenate(axis=1, name='preclassfier2')(
        [branch['branch1'][3], branch['branch2'][3], branch['branch3'][3],
         branch['branch4'][3], branch['branch5'][3], branch['branch6'][3],
         branch['branch7'][3], branch['branch8'][3], branch['branch9'][3],
         branch['branch10'][3], branch['branch11'][3], branch['branch12'][3],
         branch['branch13'][3], branch['branch14'][3], branch['branch15'][3],
         branch['branch16'][3]])
    preclassifier2 = kl.Flatten()(preclassifier2)
    preclassifier2 = kl.Dense(units=100, activation='relu')(preclassifier2)
    preclassifier2 = kl.Dense(units=100, activation='relu', name='prec2_dense')(preclassifier2)
    preclassifier2 = ArcFaceTrainable(n_classes=6, name='prec2')([preclassifier2, label])

    # tree
    tree = kl.Concatenate(axis=1, name='tree')(
        [branch['branch1'][1], branch['branch2'][1], branch['branch3'][1],
         branch['branch4'][1], branch['branch5'][1], branch['branch6'][1],
         branch['branch7'][1], branch['branch8'][1], branch['branch9'][1],
         branch['branch10'][1], branch['branch11'][1], branch['branch12'][1],
         branch['branch13'][1], branch['branch14'][1], branch['branch15'][1],
         branch['branch16'][1]])

    # preclassifier 3
    preclassifier3 = kl.Conv2D(filters=param[2], kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu')(tree)
    preclassifier3 = kl.Conv2D(filters=param[2], kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu')(
        preclassifier3)
    preclassifier3 = kl.Conv2D(filters=param[2], kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu')(
        preclassifier3)
    preclassifier3 = kl.Flatten()(preclassifier3)
    preclassifier3 = kl.Dense(units=100, activation='relu')(preclassifier3)
    preclassifier3 = kl.Dense(units=100, activation='relu', name='prec3_dense')(preclassifier3)
    preclassifier3 = ArcFaceTrainable(n_classes=6, name='prec3')([preclassifier3, label])

    # main classifier
    mainclassifier = kl.Flatten()(tree)
    mainclassifier = kl.Dense(units=100, activation='relu')(mainclassifier)
    mainclassifier = kl.Dense(units=100, activation='relu', name='mainc_dense')(mainclassifier)
    mainclassifier = ArcFaceTrainable(n_classes=6, name='mainc')([mainclassifier, label])

    # model
    model = km.Model(inputs=[branch['branch1'][0], branch['branch2'][0], branch['branch3'][0],
                             branch['branch4'][0], branch['branch5'][0], branch['branch6'][0],
                             branch['branch7'][0], branch['branch8'][0], branch['branch9'][0],
                             branch['branch10'][0], branch['branch11'][0], branch['branch12'][0],
                             branch['branch13'][0], branch['branch14'][0], branch['branch15'][0],
                             branch['branch16'][0], label],
                     outputs=[mainclassifier, preclassifier1, preclassifier2, preclassifier3])
    if summary:
        model.summary()
    model.compile(optimizer='adam',
                  loss={'prec1': 'categorical_crossentropy',
                        'prec2': 'categorical_crossentropy',
                        'prec3': 'categorical_crossentropy',
                        'mainc': 'categorical_crossentropy'},
                  loss_weights={'prec1': 0.5,
                                'prec2': 0.5,
                                'prec3': 0.5,
                                'mainc': 1.},
                  metrics=['accuracy'])
    return model


def network_B(param, sensornum, summary=False):
    def branch_create(scope):
        assert type(scope) == str, 'TypeError: scope should be a str'
        inputs = kl.Input(shape=(50, 1, 1), name=scope + '/Input')
        BN = kl.BatchNormalization(1)(inputs)
        conv1 = kl.Conv2D(filters=param[0], kernel_size=(3, 1), padding='same', strides=(1, 1), activation='elu')(BN)
        conv1 = kl.Conv2D(filters=param[0], kernel_size=(3, 1), padding='same', strides=(1, 1), activation='elu')(conv1)
        conv1 = kl.Conv2D(filters=param[0], kernel_size=(3, 1), padding='same', strides=(1, 1), activation='elu')(conv1)
        conv1 = kl.MaxPool2D(pool_size=(3, 1), strides=(3, 1), padding='same')(conv1)
        conv2 = kl.Conv2D(filters=param[1], kernel_size=(3, 1), padding='same', strides=(1, 1), activation='elu')(conv1)
        conv2 = kl.Conv2D(filters=param[1], kernel_size=(3, 1), padding='same', strides=(1, 1), activation='elu')(conv2)
        conv2 = kl.Conv2D(filters=param[1], kernel_size=(3, 1), padding='same', strides=(1, 1), activation='elu')(conv2)
        conv2 = kl.MaxPool2D(pool_size=(3, 1), strides=(3, 1), padding='same')(conv2)
        conv3 = kl.Conv2D(filters=param[2], kernel_size=(3, 1), padding='same', strides=(1, 1), activation='elu')(conv2)
        conv3 = kl.Conv2D(filters=param[2], kernel_size=(3, 1), padding='same', strides=(1, 1), activation='elu')(conv3)
        conv3 = kl.Conv2D(filters=param[2], kernel_size=(3, 1), padding='same', strides=(1, 1), activation='elu')(conv3)
        conv3 = kl.MaxPool2D(pool_size=(3, 1), strides=(3, 1), padding='same')(conv3)
        return inputs, conv2, conv1, conv3

    # label (for Arcface)
    label = kl.Input(shape=(7,), name='Arcface_input')

    # create branch
    branch = {}
    for branch_num in range(1, sensornum + 1):
        branch_name = 'branch{}'.format(branch_num)
        branch[branch_name] = branch_create(branch_name)

    # preclassifier 1
    preclassifier1 = kl.Concatenate(axis=2, name='preclassfier1')([branch['branch{}'.format(i)][2] for i in range(1, sensornum + 1)])
    preclassifier1 = kl.Flatten()(preclassifier1)
    preclassifier1 = kl.Dense(units=100, activation='relu')(preclassifier1)
    preclassifier1 = kl.Dense(units=100, activation='relu', name='prec1_dense')(preclassifier1)
    preclassifier1 = ArcFaceTrainable7(n_classes=7, name='prec1')([preclassifier1, label])

    # preclassifier 2
    preclassifier2 = kl.Concatenate(axis=2, name='preclassfier2')([branch['branch{}'.format(i)][3] for i in range(1, sensornum + 1)])
    preclassifier2 = kl.Flatten()(preclassifier2)
    preclassifier2 = kl.Dense(units=100, activation='relu')(preclassifier2)
    preclassifier2 = kl.Dense(units=100, activation='relu', name='prec2_dense')(preclassifier2)
    preclassifier2 = ArcFaceTrainable7(n_classes=7, name='prec2')([preclassifier2, label])

    # tree
    tree = kl.Concatenate(axis=2, name='tree')([branch['branch{}'.format(i)][1] for i in range(1, sensornum + 1)])

    # preclassifier 3
    preclassifier3 = kl.Conv2D(filters=param[2], kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu')(tree)
    preclassifier3 = kl.Conv2D(filters=param[2], kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu')(preclassifier3)
    preclassifier3 = kl.Conv2D(filters=param[2], kernel_size=(3, 3), padding='same', strides=(1, 1), activation='elu')(preclassifier3)
    preclassifier3 = kl.Flatten()(preclassifier3)
    preclassifier3 = kl.Dense(units=100, activation='relu')(preclassifier3)
    preclassifier3 = kl.Dense(units=100, activation='relu', name='prec3_dense')(preclassifier3)
    preclassifier3 = ArcFaceTrainable7(n_classes=7, name='prec3')([preclassifier3, label])

    # main classifier
    mainclassifier = kl.Flatten()(tree)
    mainclassifier = kl.Dense(units=100, activation='relu')(mainclassifier)
    mainclassifier = kl.Dense(units=100, activation='relu', name='mainc_dense')(mainclassifier)
    mainclassifier = ArcFaceTrainable7(n_classes=7, name='mainc')([mainclassifier, label])

    # model
    model = km.Model(inputs=[branch['branch{}'.format(i)][0] for i in range(1, sensornum + 1)] + [label],
                     outputs=[mainclassifier, preclassifier1, preclassifier2, preclassifier3])
    if summary:
        model.summary()
    model.compile(optimizer='adam',
                  loss={'prec1': 'categorical_crossentropy',
                        'prec2': 'categorical_crossentropy',
                        'prec3': 'categorical_crossentropy',
                        'mainc': 'categorical_crossentropy'},
                  loss_weights={'prec1': 0.5,
                                'prec2': 0.5,
                                'prec3': 0.5,
                                'mainc': 1.},
                  metrics=['accuracy'])
    return model


def get_output_function_A(model, layer_name):
    """
    model: 要保存的模型
    layer_name：
    """

    vector_function = K.function([model.layers[0].input, model.layers[1].input, model.layers[2].input, model.layers[3].input,
                                  model.layers[4].input, model.layers[5].input, model.layers[6].input, model.layers[7].input,
                                  model.layers[8].input, model.layers[9].input, model.layers[10].input, model.layers[11].input,
                                  model.layers[12].input, model.layers[13].input, model.layers[14].input, model.layers[15].input,
                                  model.get_layer('Arcface_input').input],
                                 [model.get_layer(layer_name).output])

    def inner(input_data):
        vector = vector_function([input_data[0][0], input_data[0][1], input_data[0][2], input_data[0][3],
                                  input_data[0][4], input_data[0][5], input_data[0][6], input_data[0][7],
                                  input_data[0][8], input_data[0][9], input_data[0][10], input_data[0][11],
                                  input_data[0][12], input_data[0][13], input_data[0][14], input_data[0][15], input_data[1]])[0]
        return vector

    return inner


def get_output_function_B(model, layer_name):
    """
    model: 要保存的模型
    layer_name：
    """

    vector_function = K.function([model.layers[i].input for i in range(21)] + [model.get_layer('Arcface_input').input],
                                 [model.get_layer(layer_name).output])

    def inner(input_data):
        vector = vector_function([input_data[0][i] for i in range(21)] + [input_data[1]])[0]
        return vector

    return inner
