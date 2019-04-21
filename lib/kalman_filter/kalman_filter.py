# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

Kalman滤波算例
"""

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


class KalmanFilter(object):
    """"""
    def __init__(self, params, true_value, obs_value):
        """初始化"""
        "参数"
        self.params = params
        self.params['sz'] = (self.params['n_iter'],)

        "初始化中间变量"
        self.x_hat = np.zeros(self.params['sz'])  # a posteri estimate of x 滤波估计值
        self.x_hat_minus = np.zeros(self.params['sz'])  # a priori estimate of x 估计值
        self.P = np.zeros(self.params['sz'])  # a posteri error estimate滤波估计协方差矩阵
        self.P_minus = np.zeros(self.params['sz'])  # a priori error estimate估计协方差矩阵
        self.K = np.zeros(self.params['sz'])  # gain or blending factor卡尔曼增益
        self.Q = self.params['Q'] # process variance

        "真值和观测值"
        self.true_value = true_value
        self.obs_value = obs_value

    def kalman_filtering(self):
        """进行Kalman滤波计算"""
        # initial guesses
        self.x_hat[0] = 0.0
        self.P[0] = 1.0

        for k in range(1, self.params['n_iter']):
            "预测"
            self.x_hat_minus[k] = self.x_hat[k - 1]  # X(k|k-1) = AX(k-1|k-1) + BU(k) + W(k),A=1,BU(k) = 0
            self.P_minus[k] = self.P[k - 1] + self.params['Q']  # P(k|k-1) = AP(k-1|k-1)A' + Q(k) ,A=1

            "更新"
            self.K[k] = self.P_minus[k] / (self.P_minus[k] + self.params['R'])  # Kg(k)=P(k|k-1)H'/[HP(k|k-1)H' + R],H=1
            self.x_hat[k] = self.x_hat_minus[k] + self.K[k] * (self.obs_value[k] - self.x_hat_minus[k])  # X(k|k) = X(k|k-1) + Kg(k)[Z(k) - HX(k|k-1)], H=1
            self.P[k] = (1 - self.K[k]) * self.P_minus[k]  # P(k|k) = (1 - Kg(k)H)P(k|k-1), H=1

    def show_results(self):
        """"""
        valid_iter = range(1, self.params['n_iter'])  # Pminus not valid at step 0

        plt.figure(figsize = [8, 4])
        sns.set_style('whitegrid')
        plt.subplot(121)
        plt.plot(self.obs_value, 'k+', markersize = 3, label = 'noisy measurements')  # obs value
        plt.plot(self.x_hat, 'b-', label = 'a posteri estimate')  # estimated value give by the filter
        plt.axhline(x, color = 'g', label = 'truth value')  # true value derivated by theoretical equations
        plt.legend()
        plt.xlabel('$Iteration$')
        plt.ylabel('$Voltage$')

        plt.subplot(122)
        plt.plot(np.array(valid_iter), self.P_minus[valid_iter], label = 'a priori error estimate')
        plt.xlabel('$Iteration$')
        plt.ylabel('$(Voltage)^2$')
        plt.show()


if __name__ == '__main__':
    # 给定参数
    params = {'n_iter': 500,
              'Q': 0.1,  # process variance
              'R': 1 ** 2}  # estimate of measurement variance

    x = -0.37727  # 真实值
    z = np.random.normal(x, 0.1, size = params['n_iter'])  # observations (normal about x, sigma=0.1)观测值

    # Kalman滤波
    self = KalmanFilter(params, true_value = x, obs_value = z)
    self.kalman_filtering()
    self.show_results()