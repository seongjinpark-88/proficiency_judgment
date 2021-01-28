import os, sys
import pickle
import pprint

from collections import defaultdict

from os.path import isdir, join
from pathlib import Path

import librosa
import librosa.display

import matplotlib.pyplot as plt
from collections import defaultdict

import pandas as pd

import numpy as np

import random

seed = 888
random.seed(seed)
np.random.seed(seed)

import tensorflow as tf

import keras
from keras import backend as K
from keras import optimizers

from sklearn.model_selection import KFold
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Model, load_model
from keras.layers import Input, Dense, Dropout, concatenate, Flatten
from keras.layers import LSTM, Bidirectional
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU

from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import r2_score

### R2
def r_squared(y_true, y_pred):
    """
    r-squared calculation
    from: https://jmlb.github.io/ml/2017/03/20/CoeffDetermination_CustomMetric4Keras/
    """
    ss_res = K.sum(K.square(y_true-y_pred))
    ss_tot = K.sum(K.square(y_true - K.mean(y_true)))
    return 1 - (ss_res / (ss_tot + K.epsilon()))
    # return r2_score(y_true, y_pred)

### Load data
phon_dict = {}
with open("data/perception_results/rhythm_v6.csv", "r") as f:
    phondata = f.readlines()
    for i in range(1, len(phondata)):
        line = phondata[i].rstrip().split(',')
        wav_name = line[0].split('.')[0]
        data = np.array(line[1:], dtype=np.float32)
        phon_dict[wav_name] = data

label_dict = {}

label_info = defaultdict(dict)
resp_files = ["accented_avgs.csv", "fluency_avgs.csv", "comp_avgs.csv"]

for resp in resp_files:
    with open(os.path.join("data/perception_results", resp) , "r") as f:
        data = f.readlines()

        for i in range(1, len(data)):
            # rating, stim_name, stim_file = data[i].rstrip().split(",")
            # print(data[i])
            stim_name, spk, rating = data[i].rstrip().split(",")
            # spk, _ = stim_name.split("_")
            if "accented" in resp:
                label_info[stim_name]['acc'] = float(rating)
            elif "fluency" in resp:
                label_info[stim_name]['flu'] = float(rating)
            else:
                label_info[stim_name]['comp'] = float(rating)

            label_info[stim_name]['spk'] = spk



dataX = []
dataY = []
for key in label_info.keys():
    acc_rating = label_info[key]['acc']
    flu_rating = label_info[key]['flu']
    comp_rating = label_info[key]['comp']
    rhythm = phon_dict[key]

    dataX.append(rhythm)
    dataY.append([acc_rating, flu_rating, comp_rating])



n_connected_units = 256
dropout = 0.2
act = 'relu'

true_labels = []
pred_labels = []
KFold = KFold(n_splits=5, shuffle=True)

for train, test in KFold.split(dataX):
    # print("Train: {0}\tTest: {1}".format(train, test))
    trainX = np.asarray([dataX[i] for i in train])
    testX = np.asarray([dataX[i] for i in test])
    trainY = np.array([dataY[i][0] for i in train])
    testY = np.asarray([dataY[i][0] for i in test])
    print(np.shape(testY))
    input_shape = (None, np.shape(trainX)[1])
    mlp_input = Input(shape=input_shape)
    output_1 = Dense(n_connected_units, activation=act)(mlp_input)
    dropout_1 = Dropout(dropout)(output_1)
    output_2 = Dense(n_connected_units, activation=act)(dropout_1)
    dropout_2 = Dropout(dropout)(output_2)
    # output_3 = Dense(n_connected_units, activation=act)(dropout_2)
    final_output = Dense(1, activation="linear")(dropout_2)

    model = Model(inputs=mlp_input, outputs=[final_output])
    opt = optimizers.Adam(learning_rate=0.0005)
    model.compile(optimizer=opt, loss="mse", metrics=['mse', r_squared])
    model.summary()

    history = model.fit(trainX, trainY,
                        epochs=1000,
                        batch_size=64,
                        shuffle=True,
                        validation_data=(testX, testY),
                        verbose=1)
    pred = model.predict(testX, batch_size=64)
    # print(pred)
    for j in range(0, len(pred)):
        # print(pred[j][0], testY[j, 0])
        true_labels.append(float(testY[j]))
        pred_labels.append(pred[j][0])
    scores = model.evaluate(testX, testY, verbose=0)
    print("CV{0} Score: {1}".format(i, scores))

print("Total r2: {0}".format(r2_score(true_labels, pred_labels)))