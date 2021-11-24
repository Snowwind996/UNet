from data import imageread
from Unet_cutted import UNet
from time import time
from torch import cuda
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch
import argparse


parser = argparse.ArgumentParser(description='Semantic Segmentation hyper parameters')
parser.add_argument('--epochnum', default=100, help='number of epoches')
parser.add_argument('--GPU', default=True, help='whether use gpu or not')
parser.add_argument('--imgnum',default=100,type=int,help='read number of image.')
parser.add_argument('--TransformNum',default=3,type=int,help='Each image will be transformed to more images by many methods')
parser.add_argument('--minibatchnum',default=1,type=int,help='Number of image in each minibatch')
parser.add_argument("--loadmodel",default=False,type=bool,help='Whether load pretrained model or not')
parser.add_argument("--model_path",default=".\\modelset\\img70_minibatch10.pth",type=str,help="It is trained model's path.")
config=parser.parse_args()
def train(config):
	epochnum,gpu=config.epochnum,config.GPU
	imgnum,transformnum=config.imgnum,config.TransformNum
	minibatchnum,loadmodel=config.minibatchnum,config.loadmodel
	model_loadpath=config.model_path
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 开启GPU加速模块
	imageset, goalset = imageread(imgnum,transformnum=transformnum)
	train_ids = TensorDataset(imageset, goalset)
	train_loader = DataLoader(dataset=train_ids,batch_size=minibatchnum, shuffle=True,pin_memory=True if gpu else False)
	print(f'imageset shape:{imageset.shape}')
	print(f'goalset.shape:{goalset.shape}')
	#读取数据，并打印数据
	time_creatnet = time()
	if gpu:
		model=UNet().to(device)
		criterion = torch.nn.BCELoss().to(device)#反向传播模块放入gpu计算。
	else:
		model=UNet()
		criterion = torch.nn.BCELoss()
	optimizer = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9, weight_decay=5e-4)
	if loadmodel:
		path=model_loadpath
		save_info = torch.load(path)
		model.load_state_dict(save_info["model"])
		#如果是进行继续训练或者迁移学习，可以这么用。
	path = 'model.pth'#加载数据完成后，对path进行修改，然后训练的参数都会在这个文件中。
	time_netfinish = time()
	print("网络创建成功！所用时间：{:.2f}秒".format(time_netfinish - time_creatnet))
	#建立网络，加载参数。
	epoch=epochnum
	print("开始进入循环！")
	for index, data in enumerate(train_loader, 1):
		imageset_batch, labelset_batch = data
		print(f"采用迷你批次图片输入输出进行计算。本次训练的是第{index}组图片")
		print("_____________________________________________________________________________________________")
		if gpu:
			imageset_batch=imageset_batch.float()
			labelset_batch=labelset_batch.float()
			imageset_batch=imageset_batch.to(device)
			labelset_batch=labelset_batch.to(device)
		mintrainloss = 1000000
		lasttrainloss=mintrainloss
		for i in range(epoch):
			count = 1
			time_begin=time()
			answer=model(imageset_batch)
			time_compute=time()
			print(f"第{i+1}轮计算已经完成！用时：{time_compute-time_begin:.3f}")
			optimizer.zero_grad()
			loss=criterion(answer,labelset_batch)
			trainloss=loss.item()
			print("The train loss is :{:.5f}".format(trainloss))
			state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': epoch}
			if trainloss < mintrainloss:
				mintrainloss=trainloss
				torch.save(state, path)
			else:
				count+=1
				if count==2:
					print(f'发生损失函数上升问题,此时的损失函数为：{trainloss:.5f},上次的损失函数为:{mintrainloss:.5f}')
					print('取消训练该涨图片，继续训练。')
					break
			loss.backward()
			cuda.synchronize()
			optimizer.step()
			time_back=time()
			print("第{}轮反向传导完成！所时：{:.3f}".format(i+1,time_back-time_compute))
			if trainloss<0.01:
				print('该次循环损失函数小于0.01，因此退出循环。')
				break
			if lasttrainloss-trainloss<0.001:
				print('该次循环的梯度下降过小，损失函数减少少于0.001，因此退出循环')
				break
	return state

save_info=train(config=config)
