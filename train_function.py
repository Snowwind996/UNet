import numpy as np
import pandas as pd
import torch

from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from time import time

import matplotlib.pyplot as plt
import time as t
from torch import cuda
class lossc():
	def __init__(self,data):
		self.data=data
	def print(self):
		plt.figure('损失函数趋势')
		plt.plot(self.data)
		print(self.data)
		plt.show()
class modelfit():
	def __init__(self,model,gpu):
		self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 开启GPU加速模块
		self.gpu=gpu
		if gpu:
			self.model=model().to(self.device)
			self.criterion=torch.nn.BCELoss().to(self.device)  # 反向传播模块放入gpu计算。
		else:
			self.model=model()
			self.criterion=torch.nn.BCELoss()
		self.optimizer = torch.optim.Adam(self.model.parameters())
	def loadmodel(self,model_loadpath):
		self.model_loadpath = model_loadpath
		path=model_loadpath
		save_info = torch.load(path)
		try:self.model.load_state_dict(save_info["model"])
		except:self.model.load_state_dict(save_info)
	def loaddata(self,datapath,labelpath,minibatchnum):
		self.dataset=torch.load(datapath).detach()
		self.goalset=torch.load(labelpath).detach()
		train_ids = TensorDataset(self.dataset, self.goalset)
		self.train_loader = DataLoader(dataset=train_ids,batch_size=minibatchnum, shuffle=True,pin_memory=True if self.gpu else False)
	def train(self,epochnum=100,show_procedure=True,eps=1e-4):
		path = 'model.pth'#加载数据完成后，对path进行修改，然后训练的参数都会在这个文件中。
		loss_set=np.array([])
		for index, data in enumerate(self.train_loader, 1):
			dataset_batch, goalset_batch = data
			print(f"采用迷你批次图片输入输出进行计算。本次训练的是第{index}组数据")
			print("_____________________________________________________________________________________________")
			if self.gpu:
				dataset_batch=dataset_batch.float()
				goalset_batch=goalset_batch.float()
				dataset_batch=dataset_batch.to(self.device)
				goalset_batch=goalset_batch.to(self.device)
			mintrainloss = 1000000
			dataset_batch=torch.tensor(dataset_batch,requires_grad=True)
			lasttrainloss=1000
			for i in range(epochnum):
				time_begin=time()
				answer=self.model(dataset_batch)
				time_fowardfinish=time()
				loss=self.criterion(answer,goalset_batch)
				self.optimizer.zero_grad()
				trainloss = loss.item()
				state = self.model.state_dict()
				if trainloss <= mintrainloss:
					mintrainloss = trainloss
					torch.save(state, path)
				loss.backward()
				time_backwardfinish=time()
				cuda.synchronize()
				self.optimizer.step()
				loss_set=np.append(loss_set,trainloss)
				if show_procedure:
					print(f"第{i}次前向传播所用时间:{(time_fowardfinish-time_begin):.2f}")
					print(f"第{i}次后向传播所用时间:{time_backwardfinish-time_fowardfinish:.2f}")
					print(f"第{i}次损失函数值为啦啦:{trainloss:.5f}")
					print(f"第{i}次运行完成的时间为:{t.strftime('%Y-%m-%d %H:%M:%S', t.localtime(t.time()))}")
				if abs(lasttrainloss-loss)<eps:
					print(f"第{i}次损失函数减少值为L{lasttrainloss-loss}，因此退出本次循环。")
					break
			self.loss_set=lossc(loss_set)
