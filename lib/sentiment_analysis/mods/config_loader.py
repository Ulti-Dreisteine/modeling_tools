# -*- coding: utf-8 -*-
"""
Created on 2019/6/13 23:01

Author: luolei

参数配置器
"""
import sys
import yaml

sys.path.append('../')


class ConfigLoader(object):
	"""配置"""
	def __init__(self):
		self.file_path = '../config/config.yml'
		self._load_yaml()

	def _load_yaml(self):
		"""加载配置文件"""
		f = open(self.file_path, 'r', encoding = 'utf-8')
		cfg = f.read()
		self.config = yaml.load(cfg)


config_loader = ConfigLoader()