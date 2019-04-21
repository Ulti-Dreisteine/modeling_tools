# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei


"""

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def show_samples(samples, color = 'b'):
	"""
	显示样本散点图
	:param samples:
	:param color:
	:return:
	"""
	plt.figure('2D Gaussian Samples')
	plt.scatter(samples[:, 0], samples[:, 1], c = color, s = 3)
	plt.grid(True)


def cov_matrix(vars, corner):
	"""
	获得协方差矩阵
	:param vars: 两个维度上的方差
	:param corner: 协方差矩阵反对角元素上的值
	:return: cov
	"""
	product = vars[0] * vars[1]
	if product < 0:
		raise ValueError('var must be non-negative')
	else:
		if (corner > pow(product, 0.5)) | (corner < -pow(product, 0.5)):
			raise ValueError('the cov matrix is not positive semidefinite')
		else:
			return np.array([[vars[0], corner], [corner, vars[1]]])


def gen_2d_gaussian(mean, cov, sample_num, show_plot = False, color = 'b'):
	"""
	生成二维高斯分布样本
	:param mean: 均值, np.array([mean_0, mean_1])
	:param cov: 协方差矩阵, np.array([[var_0, cov_0_1], [cov_1_0, var_1]])
	:param sample_num: 样本数，int
	:return: samples, np.array([[x, y],...])
	"""
	samples = np.random.multivariate_normal(mean, cov, sample_num)
	
	if show_plot == True:
		show_samples(samples, color)
	
	return samples


def cal_2d_gaussian_prob_dist_func(ort_arr, ort_mean, ort_cov):
	"""
	计算二维高斯概率密度
	:param ort_arr:
	:param mean:
	:param cov:
	:return:
	"""
	x_minus_miu = ort_arr - ort_mean
	sigma_det = ort_cov[0][0] * ort_cov[1][1]
	sigma_inv = np.linalg.inv(ort_cov)
	
	return (1 / (2 * np.pi * pow(sigma_det, 0.5))) * np.exp(-0.5 * np.dot(np.dot(x_minus_miu, sigma_inv), x_minus_miu.T))

