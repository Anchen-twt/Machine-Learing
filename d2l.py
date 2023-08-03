#@save
import collections
import hashlib
import math
import numpy as np
import os
import random
import re
import torch
import torchvision
import shutil
import pandas as pd
import requests
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.utils import data
from torchvision import transforms
from PIL import Image
from IPython import display
from matplotlib import pyplot as plt
from matplotlib_inline import backend_inline

d2l = sys.modules[__name__]

def use_svg_display():
	"""使用svg格式在Jpuyter中显示视图"""
	backend_inline.set_matplotlib_formats('svg')

def set_figsize(figsize=(3.5, 20.5)):
	"""设置matplotlib的图表大小"""
	use_svg_display()
	d2l.plt.rcParams['figure.figsize'] = figsize

def set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend):
	"""设置matplotlib的轴"""
	axes.set_xlabel(xlabel)
	axes.set_ylabel(ylabel)
	axes.set_xscale(xscale)
	axes.set_yscale(yscale)
	axes.set_xlim(xlim)
	axes.set_ylim(ylim)
	if legend:
		axes.legend(legend)
	axes.grid()
	
def plot(X, Y=None, xlabel=None, ylabel=None, legend=None, xlim=None,
         ylim=None, xscale='linear', yscale='linear',
         fmts=('-', 'm--', 'g-.', 'r:'), figsize=(3.5, 2.5), axes=None):
    """绘制数据点"""
    if legend is None:
    	legend = []
    	
    set_figsize(figsize)
    axes = axes if axes else d2l.plt.gca()
    
    # 如果X有一个轴，输出True
    def has_one_axis(X):
    	return (hasattr(X, "ndim") and X.ndim == 1 or isinstance(X, list)
    	        and not hasattr(X[0], "__len__"))
    	        
    if has_one_axis(X):
    	X = [X]
    if Y is None:
    	X, Y = [[]] * len(X), X
    elif has_one_axis(Y):
    	Y = [Y]
    if len(X) != len(Y):
    	X = X * len(Y)
    axes.cla()
    for x, y, fmt in zip(X, Y, fmts):
    	if len(X):
    		axes.plot(x, y, fmt)
    	else:
    		axes.plot(y, fmt)
    set_axes(axes, xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
    plt.show()
    
def load_array(data_arrays, batch_size, is_train=True):
	"""构造一个PyTorch数据迭代器"""
	dataset = data.TensorDataset(*data_arrays)
	return data.DataLoader(dataset, batch_size, shuffle=is_train)

def get_dataloader_workers():
	"""使用4个进程来读取数据"""
	return 4

def synthetic_data(w, b, num_examples):
	"""生成y=Xw+b+噪声"""
	X = torch.normal(0, 1, (num_examples, len(w)))
	y = torch.matmul(X, w) + b
	y += torch.normal(0, 0.01, y.shape)
	return X, y.reshape((-1, 1))  # 由行张量转化为列张量
	
def linreg(X, w, b):
	"""线性回归模型"""
	return torch.matmul(X, w) + b
	
def squared_loss(y_hat, y):
	"""均方损失"""
	return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
	
def sgd(params, lr, batch_size):
	"""小批量随机剃度下降"""
	with torch.no_grad():
		for param in params:
			param -= lr * param.grad / batch_size
			param.grad.zero_()

def get_fashion_mnist_labels(labels):
	"""返回Fashion-MNIST数据集的文本标签"""
	text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
	               'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
	return [text_labels[int(i)] for i in labels]

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
	"""绘制图像列表"""
	figsize = (num_cols * scale, num_rows * scale)
	_, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize)
	axes = axes.flatten()
	for i, (ax, img) in enumerate(zip(axes, imgs)):
		if torch.is_tensor(img):
			# 图像张量
			ax.imshow(img.numpy())
		else:
			# PIL图像
			ax.imshow(img)
		ax.axes.get_xaxis().set_visible(False)
		ax.axes.get_yaxis().set_visible(False)
		if titles:
			ax.set_title(titles[i])
	return axes

def load_data_fashion_mnist(batch_size, resize=None):
	"""下载Fashion-MNIST数据集，然后将其加载到内存中"""
	trans = [transforms.ToTensor()]
	if resize:
		trans.insert(0, transforms.Resize(resize))
	trans = transforms.Compose(trans)
	mnist_train = torchvision.datasets.FashionMNIST(
	    root="../data", train=True, transform=trans, download=True)
	mnist_test = torchvision.datasets.FashionMNIST(
        root="../data", train=False, transform=trans, download=True)
	return (data.DataLoader(mnist_train, batch_size, shuffle=True,
                            num_workers=get_dataloader_workers()),
            data.DataLoader(mnist_test, batch_size, shuffle=False, 
                            num_workers=get_dataloader_workers()))

def accuracy(y_hat, y):
	"""计算预测正确的数量"""
	if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:  # y_hat是矩阵
		y_hat = y_hat.argmax(axis=1)
	cmp = y_hat.type(y.dtype) == y
	return float(cmp.type(y.dtype).sum())

def evaluate_accuracy(net, data_iter):
    """计算在指定数据集上的精度"""
    if isinstance(net, torch.nn.Module):
    	net.eval()  # 将模型设置为评估模式
    metric = Accumulator(2)  # 正确预测数、预测总数
    with torch.no_grad():
    	for X, y in data_iter:
    		metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]   	

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一轮"""
    # 将模型设置为训练模式
    if isinstance(net, torch.nn.Module):
    	net.train()
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
    	# 计算梯度并更新参数
    	y_hat = net(X)
    	l = loss(y_hat, y)
    	if isinstance(updater, torch.optim.Optimizer):
    		# 使用PyTorch内置的优化器和损失函数
    		updater.zero_grad()
    		l.mean().backward()
    		updater.step()
    	else:
    		# 使用定制的优化器和损失函数
    		l.sum().backward()
    		updater(X.shape[0])
    	metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回训练损失和训练精度
    return metric[0] / metric[2], metric[1] / metric[2]
    		
def train_ch3(net, train_iter,  test_iter, loss, num_epochs, updater):
	"""训练模型"""
	animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
						legend=['train loss', 'train_acc', 'test_acc'])
	for epoch in range(num_epochs):
		train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
		test_acc = evaluate_accuracy(net, test_iter)
		animator.add(epoch + 1, train_metrics + (test_acc,))
	train_loss, train_acc = train_metrics
	assert train_loss < 0.5, train_loss
	assert train_acc <= 1 and train_acc > 0.7, train_acc
	assert test_acc <= 1 and test_acc > 0.7, test_acc

def predict_ch3(net, test_iter, n=6):
	"""预测标签"""
	for X, y in test_iter:
		break
	trues = get_fashion_mnist_labels(y)
	preds = get_fashion_mnist_labels(net(X).argmax(axis=1))
	titles = [true +'\n' + pred for true, pred in zip(trues, preds)]
	show_images(
		X[0:n].reshape(n, 28, 28), 1, n, titles=titles[0:n])

class Timer:
	"""记录多次运行时间"""
	def __init__(self):
		self.times = []
		self.start()
		
	def start(self):
		"""启动计时器"""
		self.tik = time.time()
		
	def stop(self):
		"""停止计时器并将时间记录在列表中"""
		self.times.append(time.time() - self.tik)
		return self.times[-1]
		
	def avg(self):
		"""返回平均时间"""
		return sum(self.times) / len(self.times)
		
	def sum(self):
		"""返回时间总和"""
		return sum(self.times)
		
	def cumsum(self):
		"""返回累计时间"""
		return np.array(self.times).cumsum(self).tolist()
		
class Accumulator:
	"""在n个变量上累加"""
	def __init__(self, n):
		self.data = [0.0] * n
	
	def add(self, *args):
		self.data = [a + float(b) for a, b in zip(self.data, args)]
		
	def reset(self):
		self.data = [0.0] * len(self.data)
		
	def __getitem__(self, idx):
		return self.data[idx]

class Animator:
	"""在动画中绘制数据"""
	def __init__(self, xlabel=None, ylabel=None, legend=None, xlim=None,
				 ylim=None, xscale='linear', yscale='linear',
				 fmts=('-', 'm--', 'g-', 'r:'), nrows=1, ncols=1,
				 figsize=(3.5, 2.5)):
        # 增量地绘制多条曲线
		if legend is None:
			legend = []
			use_svg_display()
		self.fig, self.axes = plt.subplots(nrows, ncols, figsize=figsize)
		if nrows * ncols == 1:
			self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
		self.config_axes = lambda: d2l.set_axes(
        	self.axes[0], xlabel, ylabel, xlim, ylim, xscale, yscale, legend)
		self.X, self.Y, self.fmts = None, None, fmts
        
	def add(self, x, y):
		# 向图表中添加多个数据点
		if not hasattr(y, "__len__"):
			y = [y]
		n = len(y)
		if not hasattr(x, "__len__"):
			x = [x] * n
		if not self.X:
			self.X = [[] for _ in range(n)]
		if not self.Y:
			self.Y = [[] for _ in range(n)]
		for i, (a, b) in enumerate(zip(x,y)):
			if a is not None and b is not None:
				self.X[i].append(a)
				self.Y[i].append(b)
		self.axes[0].cla()
		for x, y, fmt in zip(self.X, self.Y, self.fmts):
			self.axes[0].plot(x, y, fmt)
		self.config_axes()
		display.display(self.fig)
		display.clear_output(wait=True)
			