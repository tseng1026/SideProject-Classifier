import os
import numpy as np
import torch
from   torch.utils.data import Dataset
from   torchvision import transforms, datasets

def LoadData(train_or_test):
	import random
	
	if train_or_test == 0:
		data = datasets.CIFAR10("./data", train=True, transform=None, target_transform=None, download=True)
	
	if train_or_test == 1:
		data = datasets.CIFAR10("./data", train=False, transform=None, target_transform=None, download=True)

	return data

class DataSet(Dataset):
	def __init__(self, data, transform):
		self.data      = data
		norm_transform = transforms.Compose([
						 transforms.RandomCrop(32, padding=4),
						 transforms.RandomHorizontalFlip(),
						 transforms.ToTensor(),
						 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.4914, 0.4822, 0.4465))
						 ])
		eval_transform = transforms.Compose([
						 transforms.ToTensor(),
						 transforms.Normalize((0.4914, 0.4822, 0.4465), (0.4914, 0.4822, 0.4465))
						 ])
		if transform == "norm": self.transform = norm_transform
		if transform == "eval": self.transform = eval_transform

	def __len__(self):
		return len(self.data)

	def __getitem__(self, ind):
		img = self.data[ind][0]
		lab = self.data[ind][1]
		img = self.transform(img)
		lab = torch.tensor(lab).type(torch.LongTensor)
		return img, lab
