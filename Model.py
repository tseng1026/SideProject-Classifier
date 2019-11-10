import torch.nn as nn
import torchvision
import torchvision.models as models

class Category(nn.Module):
	def __init__(self):
		super(Category, self).__init__()

		self.conv1 = nn.Sequential(
					 nn.Conv2d(3, 6, kernel_size=5, padding=0),
					 nn.LeakyReLU(),
					 nn.MaxPool2d(2),
					 )

		self.conv2 = nn.Sequential(
					 nn.Conv2d(6, 16, kernel_size=5, padding=0),
					 nn.LeakyReLU(),
					 nn.MaxPool2d(2),
					 )

		self.last = nn.Sequential(
					nn.Linear(16*5*5, 128),
					nn.ReLU(),
					nn.Linear(128, 84),
					nn.ReLU(),
					nn.Linear(84, 10)
					)

	def forward(self, img):
		tmp = self.conv1(img)			# basic convolution layer 1
		tmp = self.conv2(tmp)			# basic convolution layer 2
		tmp = tmp.view(-1, 16*5*5)		# resize to 1 dimension
		mod = self.last(tmp)			# correspond to resultant labels
		return mod
