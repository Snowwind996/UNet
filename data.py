import os

import matplotlib.pyplot as plt
import cv2
import torch
import torchvision.transforms as transforms
from time import time
import math


def mkdir(path):
	import os
	path = path.strip()
	path = path.rstrip("\\")
	isExists = os.path.exists(path)
	if not isExists:
		os.makedirs(path)
		flag= True
	else:
		print('路径已存在')
		flag= False

def image_show(imgset,goalset):
	plt.figure()
	for i in range(imgset.shape[0]):
		shape=math.ceil(math.sqrt(2*imgset.shape[0]))
		img = imgset.detach().numpy()[i, :, :, :].transpose((1, 2, 0)).copy()
		label= goalset.detach().numpy()[i, :, :, :].transpose((1, 2, 0)).copy()
		plt.subplot(shape, shape, 2*i+1)
		plt.imshow(img, 'brg')
		plt.axis('off')
		plt.subplot(shape, shape, 2*i+2)
		plt.imshow(label, 'brg')
		plt.axis('off')
	plt.show()
def imageread(imgnum,transformnum=0,show=False):
	transform1=transforms.Compose([
	transforms.ToTensor()])
	transform2=transforms.Compose(
		[
			transforms.ToTensor(),
			transforms.Grayscale()
		]
	)
	transform_Colorjitter=transforms.Compose([
		transforms.ToTensor(),
		transforms.ColorJitter(brightness=0.3,hue=0.1,contrast=0.1,saturation=0.1)#brightness:亮度;contrast:对比度;saturation:饱和度；hue:色相偏移幅度
	])
	time_begin=time()
	print('正在读取数据...')
	for i in range(1,imgnum):
		try:
			datapath = "label\\"
			img_path=datapath+'00'+str(i)+'_json\\'+'img.png'
			label_path=datapath+"00"+str(i)+'_json\\'+'label.png'
			img=cv2.imread(img_path)
			label=cv2.imread(label_path)
			img_resize_to_size = cv2.resize(img, (512, 512))
			label_resize_to_size = cv2.resize(label, (512, 512))
			transAndUnsqueezeimg = torch.unsqueeze(transform1(img_resize_to_size), dim=0)
			transAndUnsqueezelabel = torch.unsqueeze(transform2(label_resize_to_size), dim=0)
			if i==1:
				imgset=transAndUnsqueezeimg
				goalset=transAndUnsqueezelabel
			else:
				imgset=torch.cat((imgset,transAndUnsqueezeimg),dim=0)
				goalset=torch.cat((goalset,transAndUnsqueezelabel),dim=0)
			if transformnum!=0:
				for index in range(transformnum):
						transAndUnsqueezeimg_Color=torch.unsqueeze(transform_Colorjitter(img_resize_to_size),dim=0)
						imgset = torch.cat((imgset, transAndUnsqueezeimg_Color), dim=0)
						goalset = torch.cat((goalset, transAndUnsqueezelabel), dim=0)
			goalset[goalset != 0] = 1
		except:
			continue
	time_finish=time()
	print('数据读取成功!用时{:.2f}秒'.format(time_finish-time_begin))
	print(f'imgset:{imgset.shape}')
	print(f'goalset:{goalset.shape}')
	if show:image_show(imgset,goalset)
	return imgset,goalset
if __name__=="__main__":
	imgset,goalset=imageread(imgnum=78,transformnum=0,show=True)
	print(imgset.shape)
	print(goalset.shape)
	mkpath = ".\\trainingset1"
	if mkdir(mkpath):
		os.makedirs(mkpath)
	torch.save(imgset,mkpath+"\\dataset_notransformedimageset_notransformed.pth")
	torch.save(goalset,mkpath+"\\dataset_notransformedgoalset_notransformed.pth")
