# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

寻找样本最佳的均值和协方差参数
"""


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import fsolve, minimize


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


def show_samples(samples, color = 'b'):
	""""""
	plt.figure('2D Gaussian Samples')
	plt.scatter(samples[:, 0], samples[:, 1], c = color, s = 3)
	plt.grid(True)

# TODO: this function should be checked on the param "bounds"
def ort_matrix(cov):
	"""
	计算正交变换矩阵
	:param cov:
	:return:
	"""
	a, b, c = cov[0][0], cov[0][1], cov[1][1]
	func = lambda theta: (c - a) * np.sin(theta) * np.cos(theta) + b * (pow(np.cos(theta), 2) - pow(np.sin(theta), 2))
	theta = fsolve(func, 0, xtol = 1e-10)[0]
	
	# 计算变换矩阵U
	U = np.array(
		[[np.cos(theta), -np.sin(theta)],
		 [np.sin(theta), np.cos(theta)]]
	)
	
	return U


def orthogonalize(samples, mean, cov, U, show_plot = False, color = 'r'):
	"""
	样本变换
	:param samples:
	:param mean:
	:param cov:
	:param U:
	:return:
	"""
	ort_samples = np.dot(samples, U)
	ort_cov = np.dot(np.dot(U.T, cov), U)
	ort_mean = np.dot(mean, U)
	
	if show_plot == True:
		show_samples(ort_samples, color)
	
	return ort_samples, ort_mean, ort_cov


def cal_2d_gaussian_prob_dist_func(ort_arr, ort_mean, ort_cov, eps = 1e-10):
	"""
	计算二维高斯概率密度
	:param ort_arr:
	:param mean:
	:param cov:
	:return:
	"""
	x_minus_miu = ort_arr - ort_mean
	sigma_det = ort_cov[0][0] * ort_cov[1][1] + eps
	sigma_inv = np.linalg.inv(ort_cov)
	
	return (1 / (2 * np.pi * pow(sigma_det, 0.5))) * np.exp(-0.5 * np.dot(np.dot(x_minus_miu, sigma_inv), x_minus_miu.T))


def likelihood(ort_samples, ort_mean, ort_cov):
	"""
	计算总体样本的似然概率和
	:param samples:
	:param ort_mean:
	:param ort_cov:
	:return:
	"""
	ort_samples = pd.DataFrame(ort_samples)
	ort_samples['likelihood'] = ort_samples.apply(lambda x: cal_2d_gaussian_prob_dist_func(x, ort_mean, ort_cov), axis = 1)
	
	return np.sum(ort_samples['likelihood'])


def objective(samples, mean, a, b, c):
	"""
	目标函数
	:param samples:
	:param mean:
	:param a:
	:param b:
	:param c:
	:return:
	"""
	cov = np.array([[a, b], [b, c]])
	U = ort_matrix(cov)
	ort_samples, ort_mean, ort_cov = orthogonalize(samples, mean, cov, U)
	
	return -likelihood(ort_samples, ort_mean, ort_cov) # 因为是往minimize优化，所以取负号
	

def const(eps = 1e-10):
	"""
	约束条件函数
	:param args:
	:return:
	"""
	cons = (
		{'type': 'ineq', 'fun': lambda x: x[0] * x[2] - pow(x[1], 2) - eps}, # 大于0约束
		{'type': 'ineq', 'fun': lambda x: x[0] - eps},
		{'type': 'ineq', 'fun': lambda x: x[2] - eps}
	)
	
	return cons


def find_opt_cov_params(samples, mean, x0, method, max_iter = 1000, show_plot = False):
	"""
	
	:param samples:
	:param mean:
	:param x0:
	:param cons:
	:param method:
	:param max_iter:
	:param show_plot:
	:return:
	"""
	func = lambda x: objective(samples, mean, x[0], x[1], x[2])
	cons = const()
	opt = minimize(
		func,
		x0,
		method = method,
		constraints = cons,
		tol = 1e-10,
		options = {'maxiter': max_iter}
	)
	opt_params = opt.x
	opt_cov = np.array(
		[[opt_params[0], opt_params[1]],
		 [opt_params[1], opt_params[2]]]
	)
	
	if show_plot == True:
		_ = gen_2d_gaussian(mean, opt_cov, 2000, show_plot = True, color = 'r')
	
	print('optimization %s' % opt.success)
	
	return np.array([[opt.x[0], opt.x[1]], [opt.x[1], opt.x[2]]])


if __name__ == '__main__':
	# 参数和样本
	mean = [1, 2]
	vars = [1, 3]
	corner = 1
	cov = cov_matrix(vars, corner)
	sample_num = 1000
	samples = gen_2d_gaussian(mean, cov, sample_num, show_plot = True)
	
	# 寻找最优值
	x0 = [1, 1, 2]
	cons = const()
	method = 'SLSQP'
	opt_cov = find_opt_cov_params(samples, mean, x0, max_iter = 2000, method = method, show_plot = True)
	
