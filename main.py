import numpy as np
import matplotlib.pyplot as plt
from Unet_cutted import UNet
from train_function import modelfit
import argparse
if __name__=="__main__":
    parser = argparse.ArgumentParser(description='Semantic Segmentation hyper parameters')
    parser.add_argument('--epochnum', default=100, help='number of epoches')
    parser.add_argument('--GPU', default=True, help='whether use gpu or not')
    parser.add_argument('--imgnum', default=100, type=int, help='read number of image.')
    parser.add_argument('--TransformNum', default=3, type=int,
                        help='Each image will be transformed to more images by many methods')
    parser.add_argument('--minibatchnum', default=1, type=int, help='Number of image in each minibatch')
    parser.add_argument("--loadmodel", default=False, type=bool, help='Whether load pretrained model or not')
    parser.add_argument("--model_path", default=".\\modelset\\img70_minibatch10.pth", type=str,
                        help="It is trained model's path.")
    parser.add_argument("--datapath",default=".\\dataset_notransformed\\imageset_notransformed.pth",type=str,
                        help="加载数据集的路径")
    parser.add_argument("--labelpath",default=".\\dataset_notransformed\\goalset_notransformed.pth",type=str,
                        help="加载标签的路径")
    parser.add_argument("--eps",default=1e-4,help="损失函数下降低于该值时会退出该次循环。")

    config = parser.parse_args()
    modelfit=modelfit(gpu=config.GPU,model=UNet)
    modelfit.loaddata(datapath=config.datapath,labelpath=config.labelpath,
                      minibatchnum=config.minibatchnum)
    modelfit.train(epochnum=config.epochnum,eps=config.eps)
    modelfit.loss_set.print()