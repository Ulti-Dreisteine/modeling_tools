# -*- coding: utf-8 -*-
"""
Created on Mon Feb 05 16:56:53 2018

@author: luolei

对XGBoost和LSTM模型进行K折检验
"""
from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import scipy.io as sio
import seaborn as sns
import random
import time
sns.set_style('whitegrid')

## ————————————————————————————————————————————————————————————————————————————
## functions ——————————————————————————————————————————————————————————————————
## ————————————————————————————————————————————————————————————————————————————
def MAPE(y_test, y_predicted):
    '''
    计算平均绝对百分比误差
    '''
    epsilon = 0.001
    if (type(y_test) == np.ndarray) & (type(y_predicted) == np.ndarray):
        y_test = epsilon +  y_test
        y_predicted = [p[0] for p in list(y_predicted)]
        return np.mean([abs((y_predicted[i] - y_test[i]) / y_test[i]) for i in range(len(y_test))])
    else:
        
        y_test = [p + epsilon for p in list(y_test[list(y_test.columns)[0]])]
        y_predicted = list(y_predicted)
        return np.mean([abs((y_predicted[i] - y_test[i]) / y_test[i]) for i in range(len(y_test))])
    
## ————————————————————————————————————————————————————————————————————————————
## data ———————————————————————————————————————————————————————————————————————
## ————————————————————————————————————————————————————————————————————————————
data = pd.read_csv('test_data.csv', sep = ',')[['time', 'KWH']]
data['KWH'] = 2 * (data['KWH'] - data['KWH'].min()) / (data['KWH'].max() - data['KWH'].min()) - 1
## 数据作图 
plt.figure(figsize = [6, 3])
plt.plot(data['KWH'])

## ————————————————————————————————————————————————————————————————————————————
## XGBoost 模型 K-fold检验 ————————————————————————————————————————————————————
## ————————————————————————————————————————————————————————————————————————————
## 构造样本
sample = list()
seq_len = 10
for i in range(len(data) - seq_len):
    sample.append(list(data.iloc[i : i + seq_len + 1]['KWH']))
sample = pd.DataFrame(sample)
mape = dict()
mean_mape = dict()

from xgboost.sklearn import XGBRegressor 
from sklearn.cross_validation import KFold
kf = KFold(len(sample), n_folds = 10, shuffle = True)
mape['xgboost'] = list()
for iteration, data_2 in enumerate(kf):
    train_set_index_list = list(data_2[0])
    test_set_index_list = list(data_2[1])
    train_sample = pd.DataFrame(sample.loc[train_set_index_list])
    test_sample = pd.DataFrame(sample.loc[test_set_index_list])
    
    X_train = pd.DataFrame(train_sample[list(range(seq_len))]).reset_index(drop = True)
    y_train = pd.DataFrame(train_sample[list(range(seq_len))[-1]]).reset_index(drop = True)
    X_test = pd.DataFrame(test_sample[list(range(seq_len))]).reset_index(drop = True)
    y_test = pd.DataFrame(test_sample[list(range(seq_len))[-1]]).reset_index(drop = True)
    
    xgb_model = XGBRegressor(learning_rate = 0.1,
                             n_estimators = 100,
                             seed = 0,
                             loss = 'mse')
    xgb_model.fit(X_train, y_train)
    y_predicted = xgb_model.predict(X_test)
    
    ## 计算平均绝对百分比误差
    mape['xgboost'].append(MAPE(y_test, y_predicted))
mean_mape['xgboost'] = np.mean(mape['xgboost'])

## ————————————————————————————————————————————————————————————————————————————
## LSTM 模型 K-fold检验 ———————————————————————————————————————————————————————
## ————————————————————————————————————————————————————————————————————————————
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.models import Sequential

def build_model(seq_len):
    model = Sequential()
    model.add(LSTM(1, input_dim = 1, input_length = seq_len, return_sequences = False))
    start = time.time()
    model.compile(loss = "mse", optimizer = "rmsprop")
    print("Compilation Time : ", time.time() - start)
    return model

'样本迭代次数'
epochs  = 1000
time_series = list(data['KWH'])

kf = KFold(len(sample) - seq_len, n_folds = 10, shuffle = True)
mape['LSTM'] = list()
for iteration, data_2 in enumerate(kf):
    train_set_head_num_list = data_2[0]
    test_set_head_num_list = data_2[1]
    
    train_set = list()
    for i in train_set_head_num_list:
        train_set.append(time_series[i : i + seq_len + 1])
        
    test_set = list()
    for i in test_set_head_num_list:
        test_set.append(time_series[i : i + seq_len + 1])
    
    X_train = np.array([p[: -1] for p in train_set])
    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
    y_train = np.array([p[-1] for p in train_set])
    y_train = np.reshape(y_train, (y_train.shape[0],))
    
    X_test = np.array([p[: -1] for p in test_set])
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
    y_test = np.array([p[-1] for p in test_set])
    y_test = np.reshape(y_test, (y_test.shape[0],))
    
    model = build_model(seq_len)
    model.fit(X_train, y_train, batch_size = 3000, nb_epoch = epochs)
    y_predicted = model.predict(X_test)
    
    ## 计算平均绝对百分比误差
    mape['LSTM'].append(MAPE(y_test, y_predicted))
mean_mape['LSTM'] = np.mean(mape['LSTM'])



