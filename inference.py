from Unet_cutted import UNet
import torch
from data import imageread
import math
import matplotlib.pyplot as plt
import os
import cv2
import time
from torch.utils.data import TensorDataset
from torch.utils.data import DataLoader
def test(trainmode=True,imgnum=35,model_loadpath='.\\model.pth'):
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 开启GPU加速模块
	model_path=model_loadpath
	save_info=torch.load(model_path)
	model=UNet()
	model.load_state_dict(save_info)
	model=model.to(device)
	# 加载已经保存好的模型字典。
	model.eval()
	#进入预测模式。
	if trainmode:
		time_now = time.strftime('%Y-%m-%d_%H-%M-%S')
		foldernames = ".\\" + time_now + "result"
		os.makedirs(foldernames)#根据测试日期创建文件夹。
		imgset, goalset = imageread(imgnum=imgnum, transformnum=1)
		# path_data="dataset_notransformed"
		# imgset=torch.load(path_data+"\\imageset_notransformed.pth")
		for i in range(imgset.shape[0]):
			img=torch.unsqueeze(imgset[i,:,:,:],dim=0).to(device)
			outputs=model(img).to('cpu')
			img=img.to('cpu')
			img=img.detach().numpy()[0,:,:,:].transpose((1, 2, 0)).copy()#numpy维度转化可以用transpose函数来实现。
			label=outputs.detach().numpy()[0,:,:,:].transpose((1,2,0)).copy()
			label[label>0.8]=255#概率大于0.8的点标白
			label[label<0.8]=0#概率小于0.8的点标黑。
			plt.subplot(math.ceil(math.sqrt(2*imgset.shape[0])), math.ceil(math.sqrt(2*imgset.shape[0])), 2*i+1)
			plt.imshow(img, 'brg')
			img=img*255#通过imgread读取出来的函数是被归一化过的，因此要将所有值*255，才能显现出原来的颜色。不然都接近零，显示出来的是黑色。
			cv2.imwrite(foldernames+"\\"+str(i)+"_img.jpg",img)
			plt.axis('off')
			plt.subplot(math.ceil(math.sqrt(2*imgset.shape[0])), math.ceil(math.sqrt(2*imgset.shape[0])), 2*i+2)
			plt.imshow(label,'brg')
			cv2.imwrite(foldernames+"\\"+str(i)+"_goal.jpg",label)
			plt.axis('off')
			print(f"第{i}张图片处理完成！")
		plt.savefig(foldernames + "\\summary.jpg", bbox_inches='tight')
		plt.show()
	else:
		pass
if __name__=="__main__":
	test(trainmode=True,imgnum=30)