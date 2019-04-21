# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

adam优化算法
"""
import numpy as np
import matplotlib.pyplot as plt


def func(x):
	"""
	待优化函数
	:param x:
	:return:
	"""
	y = pow(0.15 * x, 2) + np.cos(x) + np.sin(3 * x) / 3 + np.cos(5 * x) / 5 + np.sin(7 * x) / 7
	return y


def derivative_func(x):
	"""
	待优化函数导数
	:param x:
	:return:
	"""
	dy = 0.045 * x - np.sin(x) + np.cos(3 * x) - np.sin(5 * x) + np.cos(7 * x)
	return dy


def gen_samples(start_loc = -20.0, step_size = 0.1, samples_len = 401):
	"""
	生成样本
	:param show_plot:
	:param start_loc:
	:param step_size:
	:param samples_len:
	:return:
	"""
	x = np.arange(start_loc, start_loc + samples_len * step_size, step_size)
	y = func(x)
		
	return x, y


def show_optimizatin_process(results):
	"""
	显示迭代过程
	:param results:
	:return:
	"""
	plt.figure('gradient descent algorithm')
	plt.subplot(2, 1, 1)
	plt.plot(np.arange(len(results)), results)
	plt.xlim([0, len(results) - 1])
	plt.grid(True)
	plt.title('params convergence')
	plt.xlabel('iteration step')
	plt.ylabel('param value')
	plt.subplot(2, 1, 2)
	x, y = gen_samples()
	plt.plot(x, y)
	plt.xlim([np.min(x), np.max(x)])
	plt.scatter(results[:-1], [func(p) for p in results[:-1]], c = 'r', s = 20, alpha = 0.2)
	plt.scatter(results[-1], func(results[-1]), c = 'w', s = 80, marker = '*', edgecolors = 'k')
	plt.xlabel('x')
	plt.ylabel('function value')
	plt.title('iteration process')
	plt.grid(True)
	plt.tight_layout()
	

def gradient_descent(x_init = -20.0, lr = 0.1, eps = 1e-6, max_iter = 1000):
	"""
	梯度下降优化
	:param max_iter: 最大迭代次数
	:param x_init: 初始的参数值
	:param lr: 学习率
	:param eps: 收敛判别阈值
	:return:
	"""
	x = x_init
	step = 0
	results = [x]
	while True:
		x -= lr * derivative_func(x)
		results.append(x)
		
		if (len(results) > 1) & (abs(results[-1] - results[-2]) < eps):
			break
		
		step += 1
		if step == max_iter:
			break
			
	return results


def momentum_gradient_descent(x_init = -20.0, lr = 0.1, m = 0.9, eps = 1e-6, max_iter = 1000):
	"""
	动量加梯度下降法
	:param x_init:
	:param lr:
	:param m:
	:param eps:
	:param max_iter:
	:return:
	"""
	x = x_init
	step = 0
	v = 0
	results = [x]
	while True:
		v = m * v - lr * derivative_func(x)
		x += v
		results.append(x)
		
		if (len(results) > 1) & (abs(results[-1] - results[-2]) < eps):
			break

		step += 1
		if step == max_iter:
			break

	return results


def adam(x_init = -20.0, lr = 0.1, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8, eps = 1e-6, max_iteration = 1000):
	"""
	adam算法
	:param x_init:
	:param lr:
	:param beta_1:
	:param beta_2:
	:param epsilon:
	:param eps:
	:param max_iter:
	:return:
	"""
	x = x_init
	step = 0
	m = 0
	v = 0
	results = [x]
	while True:
		derivative = derivative_func(x)
		m = beta_1 * m + (1 - beta_1) * derivative
		v = beta_2 * v + (1 - beta_2) * pow(derivative, 2)
		m_hat = m / (1 - pow(beta_1, step + 1))
		v_hat = v / (1 - pow(beta_2, step + 1))
		x -= lr * m_hat / (pow(v_hat, 0.5) + epsilon)
		results.append(x)

		if (len(results) > 1) & (abs(results[-1] - results[-2]) < eps):
			break

		step += 1
		if step == max_iteration:
			break

	return results


def robust_test(lr_list, algorithms):
	"""
	算法鲁棒性测试
	:param lr_list:
	:return:
	"""
	for algorithm in algorithms:
		opt_solutions = []
		opt_results = []
		for lr in lr_list:
			opt_solution = algorithm(lr = lr, max_iter = 1e4)[-1]
			opt_solutions.append(opt_solution)
			opt_results.append(func(opt_solution))

		plt.figure('algorithm robust test', figsize = [6, 8])
		plt.subplot(2, 1, 1)
		plt.title('algorithm robust test')
		plt.plot(lr_list, opt_solutions)
		plt.xlim([np.min(lr_list), np.max(lr_list)])
		plt.legend([p.__name__ for p in algorithms])
		plt.grid(True)
		plt.xlabel('learning rate')
		plt.ylabel('optimal solution x')
		plt.subplot(2, 1, 2)
		plt.plot(lr_list, opt_results)
		plt.xlim([np.min(lr_list), np.max(lr_list)])
		plt.legend([p.__name__ for p in algorithms])
		plt.grid(True)
		plt.xlabel('learning rate')
		plt.ylabel('optimal result')
		plt.tight_layout()
	

if __name__ == '__main__':
	# x, y = gen_samples()
	
	# 算法测试 ——————————————————————————————————————————————
	# 梯度下降法
	# results = gradient_descent(lr = 1.5)
	# show_optimizatin_process(results)
	
	# 动量加梯度下降法
	# results = momentum_gradient_descent(lr = 1.5, m = 0.2)
	# show_optimizatin_process(results)
	
	# adam算法
	results = adam(lr = 4, eps = 1e-6, max_iteration = 100)
	show_optimizatin_process(results)
	
	# 算法鲁棒性测试 ————————————————————————————————————————
	# lr_list = np.arange(0.1, 10, 0.05)
	# algorithms = [gradient_descent, momentum_gradient_descent, adam]
	# robust_test(lr_list, algorithms)