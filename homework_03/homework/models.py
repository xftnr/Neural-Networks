import torch
from torch import nn

# Unlike previous classes, don't use these classes directly in train.py
# Use the functions given in main.py
# However, model definition still needs to be defined in these classes

class ScalarModel(nn.Module):
	'''
		Your code for the model that computes regression from the inputs to the scalar value of the label.
		Classification Problem (1) in the assignment
	'''

	def __init__(self):
		super().__init__()
		'''
		Your code here
		'''
		self.dim = 64*64*3
		self.fc1 = nn.Linear(self.dim,100)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(100,1)

		nn.init.constant_(self.fc2.weight, 0)
		nn.init.constant_(self.fc2.bias, 0)

	def forward(self, x):
		'''
		Input: a series of N input images x. size (N, 64*64*3)
		Output: a scalar prediction of each input image. size (N,1)
		Your code here
		'''
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)

		return x


class VectorModel(nn.Module):
	'''
		Your code for the model that computes a one-hot label. one-hot, softmax and l2 loss share the same architecture.
		All models should output the raw one-hot encoding, not the softmax!
		Classification Problem (2-4) in the assignment
	'''
	def __init__(self):
		super().__init__()
		'''
		Your code here
		'''
		self.dim = 64*64*3
		self.fc1 = nn.Linear(self.dim,100)
		self.relu = nn.ReLU()
		self.fc2 = nn.Linear(100,6)

		nn.init.constant_(self.fc2.weight, 0)
		nn.init.constant_(self.fc2.bias, 0)

	def forward(self, x):
		'''
		Input: a series of N input images x. size (N, 64*64*3)
		Output: a prediction of each input image. size (N,6)
		Your code here
		'''
		x = self.fc1(x)
		x = self.relu(x)
		x = self.fc2(x)

		return x
