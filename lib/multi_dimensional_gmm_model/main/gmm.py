# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

使用GMM算法估计二维高斯分布样本的多核参数，未加入协方差不为零的代码
"""
# TODO: 修正代码，考虑加入协方差不为零的情况

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random


def cov_matrix(vars, corner):
	"""
	获得协方差矩阵
	:param vars: 两个维度上的方差
	:param corner: 协方差矩阵反对角元素上的值
	:return: cov, 协方差矩阵
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


def show_samples(samples, color = 'b'):
	"""
	显示样本分布
	:param samples: 样本array
	:param color: 颜色, str
	:return: None
	"""
	plt.figure('2D Gaussian Samples')
	plt.scatter(samples[:, 0], samples[:, 1], c = color, s = 3)
	plt.grid(True)


def gen_rand_num(K):
	"""
	产生一个包含K个0～1的随机数list
	:param K: 核的个数
	:return 随机数array
	"""
	d = list()
	for i in range(K):
		d.append(random.random())
	return np.array(d)


def initial_params(samples, K):
	"""
	初始化分布参数
	:param samples: 二维样本，shape(N, 2)
	:param K: 设定分量数目
	:param mius: 初始K个核的均值参数array，np.array([[mean_x_0, mean_y_0], [mean_x_1, mean_y_1], ..., [mean_x_k-1, mean_y_k-1]])
	:param sigmas: 初始K个核的方差参数array，np.array([[var_x_0, var_y_0], [var_x_1, var_y_1], ..., [var_x_k-1, var_y_k-1]])
	:return pis：初始权重array, np.array([w_0, ..., w_k-1])
	"""
	samples_means = np.mean(samples, axis = 0).reshape(1, 2)
	samples_vars = np.var(samples, axis = 0).reshape(1, 2)
	mius = np.dot(np.ones(K).reshape(K, 1), samples_means) + np.dot(gen_rand_num(K).reshape(K, 1), samples_means)
	sigmas = np.dot(np.ones(K).reshape(K, 1), samples_vars)
	pis = gen_rand_num(K)
	
	return mius, sigmas, pis


def cal_2d_gaussian_prob_dist_func(ort_arr, ort_mean, ort_cov, eps = 1e-10):
	"""
	计算单个样本点的二维高斯概率密度
	:param ort_arr:
	:param mean:
	:param cov:
	:return:
	"""
	x_minus_miu = ort_arr - ort_mean
	sigma_det = ort_cov[0][0] * ort_cov[1][1] + eps
	sigma_inv = np.linalg.inv(ort_cov)
	
	return (1 / (2 * np.pi * pow(sigma_det, 0.5))) * np.exp(-0.5 * np.dot(np.dot(x_minus_miu, sigma_inv), x_minus_miu.T))


def samples_gaussian_pdf(ort_samples, ort_mean, ort_cov):
	"""
	计算样本中所有点各自的概率密度
	:param samples:
	:param ort_mean:
	:param ort_cov:
	:return:
	"""
	ort_samples = pd.DataFrame(ort_samples)
	ort_samples['likelihood'] = ort_samples.apply(lambda x: cal_2d_gaussian_prob_dist_func(x, ort_mean, ort_cov), axis = 1)
	
	return np.array(ort_samples['likelihood'])


def cal_gaussian_pdf_values(samples, mius, sigmas):
	"""
	计算K类中各自的样本对应的概率密度
	:param samples: 样本
	:param mius: K个核的均值参数array，np.array([[mean_x_0, mean_y_0], [mean_x_1, mean_y_1], ..., [mean_x_k-1, mean_y_k-1]])
	:param sigmas: K个核的方差参数array，np.array([[var_x_0, var_y_0], [var_x_1, var_y_1], ..., [var_x_k-1, var_y_k-1]])
	:return:
	"""
	K = len(mius)
	gaussian_pdf_values = []
	for k in range(K):
		cov = np.array(
			[[sigmas[k][0], 0],
			 [0, sigmas[k][1]]]
		)
		gaussian_pdf_values.append(samples_gaussian_pdf(samples, mius[k], cov))
	
	return gaussian_pdf_values


def cal_weighed_gaussian_pdf_value(gaussian_pdf_values, pis):
	"""
	计算K类加权后的各样本概率密度
	:param gaussian_pdf_values: 原样本各点概率密度
	:param pis: 权重
	:return: K类加权后的各样本概率密度
	"""
	K = len(gaussian_pdf_values)
	gaussian_pdf_values_weighted = []
	for k in range(K):
		gaussian_pdf_values_weighted.append(np.array([pis[k] * p for p in gaussian_pdf_values[k]]))
	
	return gaussian_pdf_values_weighted


def cal_gammas(samples, gaussian_pdf_values_weighted):
	"""
	归一化每个样本在各类的权重
	:param samples: 样本array
	:param gaussian_pdf_values_weighted: 加权后的高斯概率密度值
	:return:
	"""
	K = len(gaussian_pdf_values_weighted)
	N = len(samples)
	gammas = []  # gammas有K个元素list，每个元素与samples等长
	gaussian_pdf_weighted_sum = [sum(gaussian_pdf_values_weighted[i][j] for i in range(K)) for j in range(N)]  # 单样本三类中的总和
	for k in range(K):
		gammas.append([gaussian_pdf_values_weighted[k][i] / gaussian_pdf_weighted_sum[i] for i in range(N)])
	
	return gammas


def em_iteration(samples, mius, sigmas, pis, max_iter = 1000, tol = 1e-6, show_plot = False):
	"""
	进行EM迭代
	:param samples: 二维样本array, np.array([[x, y], ...])
	:param mius: K个核的均值参数array，np.array([[mean_x_0, mean_y_0], [mean_x_1, mean_y_1], ..., [mean_x_k-1, mean_y_k-1]])
	:param sigmas: K个核的方差参数array，np.array([[var_x_0, var_y_0], [var_x_1, var_y_1], ..., [var_x_k-1, var_y_k-1]])
	:param pis: K个核的权重array, np.array([w_0, ..., w_k-1])
	:param max_iter: 最大迭代次数
	:param tol: 误差限
	:param show_plot: 是否显示按照估计的参数得到的散点图
	:return:
	"""
	K = len(mius)
	N = len(samples)
	max_log_likelihood = []
	for iteration in range(max_iter):
		# E step: 计算gamma
		gaussian_pdf_values = cal_gaussian_pdf_values(samples, mius, sigmas)
		gaussian_pdf_values_weighted = cal_weighed_gaussian_pdf_value(gaussian_pdf_values, pis)
		gammas = cal_gammas(samples, gaussian_pdf_values_weighted)
		
		# M step：参数更新
		# 1. miu更新
		for k in range(K):
			Nk = np.sum(gammas[k])  # 在k类上的gamma之和
			mius[k] = sum([gammas[k][i] * samples[i] for i in range(N)]) / Nk
		
		# 2. sigma更新
		for k in range(K):
			Nk = np.sum(gammas[k])
			sigmas[k] = sum([gammas[k][i] * pow((samples[i] - mius[k]), 2) for i in range(N)]) / Nk
		
		# 3. pi更新
		for k in range(K):
			Nk = np.sum(gammas[k])
			pis[k] = Nk / N
		
		# 计算对数似然函数
		max_log_likelihood.append(
			np.log(
				np.sum(
					np.dot(pis.reshape(1, K), np.array(cal_gaussian_pdf_values(samples, mius, sigmas)).reshape(K, N))
				)
			)
		)
		
		print('iter step %s, max_log_likelihood: %s' % (iteration, max_log_likelihood[-1]))
		
		if (iteration >= 1) & (abs(max_log_likelihood[iteration] - max_log_likelihood[iteration - 1]) <= tol):
			if show_plot == True:
				for k in range(K):
					cov = cov_matrix(sigmas[k], 0)
					_ = gen_2d_gaussian(mius[k], cov, sample_num, show_plot = True, color = 'r')
			break
		elif iteration == max_iter - 1:
			print('GMM failed')
	
	return mius, sigmas, pis


if __name__ == '__main__':
	# 生成参数和样本
	sample_num = 300
	means = [[1, 2], [12, 2], [6, 12]]
	vars = [[1, 3], [2, 3], [3, 2]]
	corners = [0, 0, 0]
	for i in range(len(means)):
		cov = cov_matrix(vars[i], corners[i])
		if i == 0:
			samples = gen_2d_gaussian(means[i], cov, sample_num, show_plot = True)
		else:
			samples = np.vstack((samples, gen_2d_gaussian(means[i], cov, sample_num, show_plot = True)))
	
	# 初始化计算参数
	K = 5  # kernel数
	mius, sigmas, pis = initial_params(samples, K)  # 初始的均值、方差和kernel分布权重
	
	# 进行最大期望迭代
	mius, sigmas, pis = em_iteration(samples, mius, sigmas, pis, max_iter = 1000, tol = 1e-5, show_plot = True)
	

	

	
	


