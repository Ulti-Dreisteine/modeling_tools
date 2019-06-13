# -*- coding: utf-8 -*-
"""
Created on 2019/6/13 22:30

Author: luolei

数据提取
"""
import numpy as np
import pandas as pd
import re


def extract_hanzi(df):
	"""提取汉字"""
	pattern = re.compile(r'[\u4e00-\u9fa5]+')  # 提取汉字
	return pattern.findall(df)


def extract_labels(data):
	""""""
	max_len = int(data.length.quantile(0.75))
	features = [i + ['未知词汇'] * max_len for i in data['words']]
	features = [i[: max_len] for i in features]
	labels = data['label'].values.tolist()
	return features, labels


def build_charactors_dict(features):
	"""构建词汇字典"""
	charactors = []
	for i in features:
		charactors.extend(i)
	charactors = set(charactors)
	chara2idx_dict = {word: i + 1 for i, word in enumerate(charactors)}
	chara2idx_dict['未知词汇'] = 0
	idx2chara_dict = {i + 1: word for i, word in enumerate(charactors)}
	idx2chara_dict[0] = '未知词汇'
	return idx2chara_dict, chara2idx_dict


def list_keys_values(chara2idx_dict, list):
	return [chara2idx_dict[i] for i in list]


def build_samples_and_targets():
	"""构建样本集和目标数据集"""
	# 载入数据
	data = pd.read_excel('../tmp/comments.xls')
	data['words'] = data.loc[:, 'words'].apply(lambda x: extract_hanzi(x))

	# 提取特征和标签
	features, labels = extract_labels(data)

	# 构建词汇字典
	idx2chara_dict, chara2idx_dict = build_charactors_dict(features)

	# 句子样本向量
	sentences = np.array([list_keys_values(chara2idx_dict, p) for p in features])
	sentences = sentences / 11000  # 注意归一化

	# 样本和目标数据集
	samples, targets = sentences, np.array(labels).reshape(-1, 1)

	print(
		'\nsamples shape: {}, max_value: {}, min_value: {}'.format(
			sentences.shape, np.max(sentences), np.min(sentences)
		)
	)
	print('targets shape: {}'.format(targets.shape))

	return samples, targets


if __name__ == '__main__':
	samples, targets = build_samples_and_targets()
