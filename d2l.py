#@save
import collections
import hashlib
import math
import numpy as np
import os
import random
import re
import torch
from torch.utils import data
import shutil
import sys
import tarfile
import time
import zipfile
from collections import defaultdict
from IPython import display
from matplotlib import pyplot as plt

d2l = sys.modules[__name__]

def set_figsize(figsize=(3.5, 20.5)):
	"""设置matplotlib的图表大小"""
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

def load_array(data_arrays, batch_size, is_train=True):
	"""构造一个PyTorch数据迭代器"""
	dataset = data.TensorDataset(*data_arrays)
	return data.DataLoader(dataset, batch_size, shuffle=is_train)

def synthetic_data(w, b, num_examples):
	"""生成y=Xw+b+噪声"""
	X = torch.normal(0, 1, (num_examples, len(w)))
	y = torch.matmul(X, w) + b
	y += torch.normal(0, 0.01, y.shape)
	return X, y.reshape((-1, 1))
	
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