# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

lstm模型预测电量变化
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math


def mape(y_test, y_predicted):
    """
    计算平均绝对百分比误差
    """
    epsilon = 0.001
    if (type(y_test) == np.ndarray) & (type(y_predicted) == np.ndarray):
        y_test = epsilon + y_test
        y_predicted = [p[0] for p in list(y_predicted)]
        return np.mean([abs((y_predicted[i] - y_test[i]) / y_test[i]) for i in range(len(y_test))])
    else:
        y_test = [p + epsilon for p in list(y_test[list(y_test.columns)[0]])]
        y_predicted = list(y_predicted)
        return np.mean([abs((y_predicted[i] - y_test[i]) / y_test[i]) for i in range(len(y_test))])


class EnergyConsumptionPrediction(object):
    """
    电量消耗预测
    """

    def __init__(self, dataframe, X_seq_len = 5, pred_horizon = 1):
        """
        初始化
        :param dataframe: 数据时间变化表
        :param X_seq_len: 取下来的X序列长度
        :param pred_horizon: 预测窗口长度
        """
        self.dataframe = dataframe
        self.X_seq_len = X_seq_len
        self.pred_horizon = pred_horizon
        self.change_time_values()
        self.normalize_data_values()

    def change_time_values(self):
        """
        转换时间格式
        :param dataframe:
        :return:
        """
        import datetime

        start_time = self.dataframe.iloc[0]['time']
        start_time = datetime.datetime.strptime(start_time, '%Y-%m-%d %H:%M:%S')

        def cal_delta_time(time):
            """
            calculate delta time
            :param time:
            :return:
            """
            new_time = datetime.datetime.strptime(time, '%Y-%m-%d %H:%M:%S')
            return (new_time - start_time).seconds / 3600 + (new_time - start_time).days * 24

        self.dataframe['time'] = self.dataframe['time'].apply(lambda time: cal_delta_time(time))

    def normalize_data_values(self, up_bound = 5000):
        """
        KWH数据归一化
        :return:
        """
        self.dataframe['KWH'] = self.dataframe['KWH'].apply(lambda x: x / up_bound)

    def get_sample_and_label(self, test_ratio = 0.3):
        """
        获取样本, 待预测值在最后一个元素位置
        :param: X_seq_len: 预测样本的X
        :param: pred_horizon: 预测值距离
        :return:
        """
        time_series = list(self.dataframe['KWH'])
        sample = []
        for i in range(len(time_series)):
            sample.append(time_series[i: i + self.X_seq_len + self.pred_horizon])
            if i + self.X_seq_len + self.pred_horizon > len(time_series):
                break
        self.sample = sample

        from sklearn.utils import shuffle
        self.sample = shuffle(self.sample)

        n_sample = len(self.sample)
        self.train_sample = self.sample[: math.floor(n_sample * (1 - test_ratio))]
        self.test_sample = self.sample[math.floor(n_sample * (1 - test_ratio)):]

        return self.train_sample, self.test_sample

    def train_LSTM_model(self, epochs = 1000, show_plots = False):
        """
        训练LSTM模型
        :return:
        """
        from keras.layers.recurrent import LSTM
        from keras.models import Sequential
        import time

        X_train = np.array([p[0: self.X_seq_len] for p in self.train_sample])
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        y_train = np.array([p[-1] for p in self.train_sample])
        y_train = np.reshape(y_train, (y_train.shape[0], 1))

        X_test = np.array([p[0: self.X_seq_len] for p in self.test_sample])
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
        y_test = np.array([p[-1] for p in self.test_sample])
        y_test = np.reshape(y_test, (y_test.shape[0],))

        model = Sequential()
        model.add(LSTM(1, input_dim = 1, input_length = self.X_seq_len, return_sequences = False))
        start = time.time()
        model.compile(loss = "mse", optimizer = "rmsprop")
        print("LSTM Compilation Time : ", time.time() - start)

        model.fit(X_train, y_train, batch_size = 3000, nb_epoch = epochs)
        y_predicted = model.predict(X_test)

        if show_plots == True:
            plt.figure(figsize = [6, 4])
            plt.plot(y_test)
            plt.hold(True)
            plt.plot(y_predicted)
            plt.grid(True)
            plt.legend(['true value', 'predicted value'], loc = 4)
            plt.tight_layout()

        return mape(y_test, y_predicted)


if __name__ == '__main__':
    # LOAD DATA
    selected_columns = ['time', 'KWH']
    data = pd.read_csv('test_data.csv')[selected_columns]

    # PROCESS DATA
    self = EnergyConsumptionPrediction(data, X_seq_len = 20, pred_horizon = 1)
    train_sample, test_sample = self.get_sample_and_label()
    mape_values = self.train_LSTM_model(show_plots = True)
