# -*- coding: utf-8 -*-
"""
Created on Mon Oct  8 14:32:52 2018

@author: luolei

通过扩散模型生成插值结果
"""
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import copy
from matplotlib.colors import LogNorm
from pylab import hist2d
import sys

sys.path.append('../')


class Particle(object):
	"""
	定义二维平面颗粒
	"""

	def __init__(self, K, dt):
		"""
		初始化
		:param K: 各方向的随机扰动系数
		:param dt: 时间差
		"""
		self.K = K
		self.dt = dt

	def get_vel(self, vel):
		"""
		获取速度
		:param vel: [vel_x, vel_y], 速度向量
		"""
		self.vel = vel

	def get_loc(self, loc):
		"""
		获取位置
		:param loc: [loc_x, loc_y], 位置向量
		"""
		self.loc = loc

	def cal_random_pulse(self, miu = 0.0, std_var = 1.0):
		"""
		计算随机扰动
		:param miu: float, 随机扰动速度均值
		:param std_var: float, 随机扰动速度标准差
		"""
		r = np.random.normal(miu, std_var, 2)
		self.random_pulse = r * pow(2 * self.K * self.dt, 0.5)

	def update_loc(self):
		"""
		更新颗粒位置
		"""
		self.loc = self.loc + self.vel * self.dt + self.random_pulse


class LagrangeSimulationForParticles(object):
	"""
	颗粒的随机游走过程模拟
	"""
	def __init__(self, particles_num_generated, simulation_steps = 100, dt = 0.01):
		"""
		初始化
		:param particles_num_generated: int, 初始颗粒数
		:param simulation_steps: int, 模拟的步数
		:param dt: int, 模拟的时间步长
		"""
		self.particles_num_generated = particles_num_generated
		self.simulation_steps = simulation_steps
		self.dt = dt

		self.particles_information = pd.DataFrame(
			list(range(self.particles_num_generated)),
			columns = ['particle_index']
		)
		self.particles_information['particle_information'] = self.particles_information.apply(
			lambda x: Particle(K = np.array([1.0, 1.0]), dt = self.dt),
			axis = 1
		)

	@staticmethod
	def get_particle_vel(particle, vel):
		particle.get_vel(vel)
		return particle

	@staticmethod
	def get_particle_loc(particle, loc):
		particle.get_loc(loc)
		return particle

	@staticmethod
	def cal_particle_random_pulse(particle, miu = 0.0, std_var = 1.0):
		particle.cal_random_pulse(miu, std_var)
		return particle

	@staticmethod
	def update_particle_loc(particle):
		particle.update_loc()
		return particle

	def init_vels_and_locs(self, init_vels, init_locs):
		"""
		通过有放回抽样初始化每个质点的速度和位置
		:param init_vels: [[vel_0_x, vel_0_y], [vel_1_x, vel_1_y], ...], 初始的速度候选序列list
		:param init_locs: [[loc_0_x, loc_0_y], [loc_1_x, loc_1_y], ...], 初始的位置候选序列list
		"""
		self.particles_information['particle_information'] = self.particles_information[
			'particle_information'].apply(
			lambda particle: self.get_particle_vel(particle, random.choice(init_vels))
		)

		self.particles_information['particle_information'] = self.particles_information[
			'particle_information'].apply(
			lambda particle: self.get_particle_loc(particle, random.choice(init_locs))
		)

	def generate_wind_vels(self, wind_vels = None):
		"""
		生成风速数据
		"""
		if wind_vels is None:
			wind_vels = [1.0, 1.0]

		self.wind_vels_list = np.array(
			self.simulation_steps * wind_vels
		).reshape(-1, 2)

	@staticmethod
	def normalize(array, length = 1.0):
		"""
		归一化向量
		:param length: float, 归一化后的向量长度
		:param array: array, 待归一化的向量
		:return:array: array, 归一化后的向量
		"""
		original_length = np.linalg.norm(array, 2)
		v = array * length / original_length
		return v

	def run_simulation(self, random_walk_miu = 0.0, random_walk_std_var = 1.0, local_emissions_per_step = 0, show_plot = True, **kwargs):
		"""
		进行模拟
		:param local_emissions_per_step: int, 每个step内新增排放颗粒数
		:param random_walk_miu: float, 随机脉动的均值
		:param random_walk_std_var: float, 随机脉动的方差
		:param show_plot: bool, 是否显示模拟动态图像
		:return:particle_infomation_list: [[particle_0_t0, particle_1_t0, ...], [particle_0_t1, particle_1_t1, ...], ...], 长度为模拟步数的模拟颗粒信息总体列表, 里面每个子list里保存了该时刻所有颗粒类信息
		"""
		if show_plot:
			plt.figure('random_walk_particle_diffusion_simulation', figsize = [4.5, 4])

		particle_information_list = []

		step = 0
		while step < self.simulation_steps:
			print('performing simulation: step %s' % step)

			# 获得颗粒位置
			self.particles_information['particle_loc'] = self.particles_information[
				'particle_information'].apply(
				lambda particle: particle.loc
			)

			if show_plot:
				particle_locs = list(self.particles_information['particle_loc'])
				particle_locs_on_x = [p[0] for p in particle_locs]
				particle_locs_on_y = [p[1] for p in particle_locs]

				plt.clf()

				hist2d(particle_locs_on_x,
					   particle_locs_on_y,
					   range = [[-100, 100], [-100, 100]],
					   bins = 500,
					   norm = LogNorm())
				plt.colorbar()
				plt.scatter(0, 0, c = 'r', s = 2)
				plt.grid(True)
				plt.xlim([-100, 100])
				plt.ylim([-100, 100])
				plt.xlabel('loc x')
				plt.ylabel('loc y')
				plt.legend(['step = %s' % step], loc = 'upper right')
				direc = self.normalize(self.wind_vels_list[step], 15)
				plt.arrow(50, -50,
						  direc[0], direc[1],
						  head_width = 5,
						  shape = 'full',
						  linewidth = 1.5)
				plt.tight_layout()
				plt.show()
				plt.pause(1.0)

			# 计算随机脉动位移
			self.particles_information['particle_information'] = self.particles_information[
				'particle_information'].apply(
				lambda particle: self.cal_particle_random_pulse(particle,
													   miu = random_walk_miu,
													   std_var = random_walk_std_var)
			)

			# 获得实时风速
			self.particles_information['particle_information'] = self.particles_information[
				'particle_information'].apply(
				lambda particle: self.get_particle_vel(particle, self.wind_vels_list[step])
			)

			# 更新颗粒位置
			self.particles_information['particle_information'] = self.particles_information[
				'particle_information'].apply(
				lambda particle: self.update_particle_loc(particle)
			)

			if local_emissions_per_step > 0:
				local_emission_locs = kwargs['local_emission_locs']
				particles_generated = pd.DataFrame([Particle(K = np.array([1.0, 1.0]), dt = self.dt) for i in range(local_emissions_per_step)], columns = ['particle_information'])
				particles_generated['particle_index'] = particles_generated.index
				particles_generated['particle_information'] = particles_generated[
					'particle_information'].apply(
					lambda particle: self.get_particle_vel(particle, random.choice(np.array([[0, 0]])))
				)

				particles_generated['particle_information'] = particles_generated[
					'particle_information'].apply(
					lambda particle: self.get_particle_loc(particle, random.choice(local_emission_locs))
				)

				particles_generated['particle_loc'] = particles_generated[
					'particle_information'].apply(
					lambda particle: particle.loc
				)

				self.particles_information = pd.concat([self.particles_information, particles_generated[['particle_index', 'particle_information', 'particle_loc']]], axis = 0)
				self.particles_information.reset_index(drop = True, inplace = True)
				self.particles_information['particle_index'] = self.particles_information.index

			particle_information_list.append(copy.deepcopy(list(self.particles_information['particle_loc'])))  # step = 0时已经有一次排放

			step += 1

		return particle_information_list


def particles_effusion_simulation(particles_num = 5000, simulation_steps = 20, dt = 1, init_vels = None, init_locs = None, wind_vels = None, random_walk_miu = 0.0, random_walk_std_var = 0.0,
								  show_plot = False, local_emissions_per_step = 50, local_emission_locs = None):
	"""
	生成颗粒扩散模拟数据
	:param local_emission_locs: np.array([[loc_0_x, loc_0_y], [loc_1_x, loc_1_y], ...]), 当地排放点位置
	:param local_emissions_per_step: int, 当地排放点在一个dt内总共排放的粒子数
	:param particles_num: int, 初始模拟粒子数
	:param simulation_steps: int, 模拟步数
	:param dt: int, 模拟时间间隔
	:param init_vels: np.array([vel_x, vel_y]), 颗粒的初始速度
	:param init_locs: np.array([[loc_x_0, loc_y_0], [loc_x_1, loc_y_1], ...]), 模拟开始时已有颗粒初始位置
	:param wind_vels: [vel_x, vel_y], 风速
	:param random_walk_miu: float, 随机游走速度的均值
	:param random_walk_std_var: float, 随机游走速度的标准差
	:param show_plot: bool, 是否显示模拟动态图
	:return: particle_information_list: [[particle_0_t0, particle_1_t0, ...], [particle_0_t1, particle_1_t1, ...], ...], 长度为模拟步数的模拟颗粒信息总体列表, 里面每个子list里保存了该时刻所有颗粒类信息
	"""
	if init_vels is None:
		init_vels = np.array([[1, 1]])
	if init_locs is None:
		init_locs = 50 * np.random.random((100, 2)) - 25
	if wind_vels is None:
		wind_vels = [[1.0, 1.0]]
	if local_emission_locs is None:
		local_emission_locs = 4 * np.random.random((100, 2)) - 2

	lsp = LagrangeSimulationForParticles(
		particles_num_generated = particles_num,
		simulation_steps = simulation_steps,
		dt = dt
	)
	lsp.init_vels_and_locs(
		init_vels = init_vels,
		init_locs = init_locs
	)
	lsp.generate_wind_vels(wind_vels)

	particle_information_list = lsp.run_simulation(
		random_walk_miu = random_walk_miu, random_walk_std_var = random_walk_std_var, show_plot = show_plot, local_emissions_per_step = local_emissions_per_step, local_emission_locs = local_emission_locs
	)

	return particle_information_list


