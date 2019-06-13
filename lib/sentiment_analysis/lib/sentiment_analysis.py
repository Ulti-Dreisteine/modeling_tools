# -*- coding: utf-8 -*-
"""
Created on 2019/6/13 22:27

Author: luolei

语义分析
"""
import torch
import torch.utils.data as Data
from torch.utils.data import DataLoader
import json
import numpy as np
import sys

sys.path.append('../')

from mods.build_samples_and_targets import build_samples_and_targets
from mods.config_loader import config_loader
from mods.model import LSTM, initialize_lstm_params
from mods.loss_criterion import criterion


def save_model_state_and_params(lstm):
	"""保存模型状态和参数"""
	# 保存模型文件
	torch.save(lstm.state_dict(), '../tmp/lstm_state_dict.pth')

	# 保存模型结构参数
	model_struc_params = {'lstm': {'input_size': lstm.lstm.input_size}}
	with open('../tmp/model_struc_params.pkl', 'w') as f:
		json.dump(model_struc_params, f)


if __name__ == '__main__':
	# 模型训练参数
	lr = config_loader.config['model_params']['lr']
	epochs = config_loader.config['model_params']['epochs']
	batch_size = config_loader.config['model_params']['batch_size']
	use_cuda = config_loader.config['model_params']['train_use_cuda']

	# 构建样本并扩增维数
	samples, targets = build_samples_and_targets()
	samples, targets = np.expand_dims(samples, axis = 2), np.expand_dims(targets, axis = 2)
	print('\nsamples shape: {}'.format(samples.shape))
	print('targets shape: {}'.format(targets.shape))

	# 构建训练集
	X_train = samples.astype(np.float32)
	y_train = targets.astype(np.float32)

	torch_dataset = Data.TensorDataset(torch.from_numpy(X_train), torch.from_numpy(y_train))
	trainloader = DataLoader(torch_dataset, batch_size = batch_size, shuffle = True)

	# 初始化模型
	input_size = X_train.shape[2]
	lstm = LSTM(input_size)
	lstm = initialize_lstm_params(lstm)

	# 设定优化器
	optimizer = torch.optim.Adam(
		lstm.parameters(),
		lr = lr
	)

	# cuda设置
	if use_cuda:
		torch.cuda.empty_cache()
		trainloader = [(train_x.cuda(), train_y.cuda()) for (train_x, train_y) in trainloader]
		lstm = lstm.cuda()

	# 进行模型训练
	loss_record = []
	for epoch in range(epochs):
		for train_x, train_y in trainloader:
			lstm_out = lstm(train_x)
			lstm_out = lstm_out[:, -1:, 0]																				# 取最后一个输出，也可以再往后预测一位取输出
			loss = criterion(lstm_out, train_y[:, :, 0])
			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		loss_record.append(loss)
		print('epoch {}， loss {:.6f}'.format(epoch, loss))

		if epoch % 100 == 0:
			save_model_state_and_params(lstm)

	# 损失函数记录
	loss_record = [float(p.detach().cpu().numpy()) for p in loss_record]

	# 保存损失函数记录
	with open('../tmp/train_loss.pkl', 'w') as f:
		json.dump(loss_record, f)

	# 保存模型文件
	save_model_state_and_params(lstm)





