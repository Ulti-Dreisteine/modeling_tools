# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

keras神经网络算例
"""
import keras
from keras.models import Sequential
from keras.layers import Dense
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def gen_cluster(samples_len, center_loc = None):
	"""
	生成聚团样本
	:param center_loc:
	:param samples_len: 样本长度
	:return:
	"""
	if center_loc is None:
		center_loc = [0, 0]
	
	samples = pd.DataFrame(np.random.normal(center_loc, [1, 1], (samples_len, 2)))
	samples['index'] = samples.index
	return samples
	
	
def gen_circle(samples_len, center_loc = None):
	"""
	生成圈状样本
	:param samples_len:
	:param center_loc:
	:return:
	"""
	if center_loc is None:
		center_loc = [0, 0]
	
	x = 2.0 * np.pi * np.random.random((samples_len, 1))
	samples = np.hstack(
		(
			5 * np.sin(x),
			5 * np.cos(x)
		),
	) + center_loc
	samples = samples + 1 * np.random.random((samples_len, 2))
	samples = pd.DataFrame(samples)
	samples['index'] = samples.index
	return samples


def gen_samples(samples_len, show_plot = False):
	"""
	生成样本
	:param samples_len:
	:param center_loc:
	:return:
	"""
	cluster = gen_cluster(round(samples_len / 2))
	cluster['label'] = cluster.loc[:, 'index'].apply(lambda x: 0)
	circle = gen_circle(round(samples_len / 2))
	circle['label'] = circle.loc[:, 'index'].apply(lambda x: 1)
	samples = pd.concat([cluster, circle], axis = 0)
	
	if show_plot:
		plt.figure('distribution of samples')
		plt.scatter(samples[0], samples[1])
		plt.grid(True)
		
	return samples

	
if __name__ == '__main__':
	# 生成样本
	raw_samples = gen_samples(samples_len = 1000, show_plot = False)
	
	# 输入训练数据 keras接收numpy数组类型的数据
	samples = np.array(np.array(raw_samples[[0, 1]])).reshape(1, 2, -1)  # TODO：调整数据输入情况
	labels = np.array(raw_samples['label']).reshape(1, -1)
	
	# 使用序贯模型
	model = Sequential()
	model.add(Dense(2, input_shape = (2, ), activation = 'tanh'))
	model.compile(optimizer = 'sgd', loss = 'mse')
	
	model.fit(samples, labels, epochs = 2000)
	
	# N = 100  # number of points per class
	# D = 2  # dimensionality
	# K = 3  # number of classes
	# X = np.zeros((N * K, D))  # data matrix (each row = single example)
	# y = np.zeros(N * K, dtype = 'uint8')  # class labels
	# for j in range(K):
	# 	ix = list(range(N * j, N * (j + 1)))
	# 	r = np.linspace(0.0, 1, N)  # radius
	# 	t = np.linspace(j * 4, (j + 1) * 4, N) + np.random.randn(N) * 0.2  # theta
	# 	X[ix] = np.c_[r * np.sin(t), r * np.cos(t)]
	# 	y[ix] = j  # 打标签
	#
	# # 将y转化为one-hot编码
	# # y = (np.arange(K) == y[:,None]).astype(int)
	# y = np.eye(K)[y]