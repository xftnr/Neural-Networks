import torch.nn as nn

class MainLinear(nn.Module):
	def __init__(self):
		super(MainLinear, self).__init__()
		'''
		Your code here
		'''
		self.function = nn.Linear(2, 2)


	def forward(self, x):
		'''
		Your code here
		'''
		return self.function(x)

class MainDeep(nn.Module):
	def __init__(self):
		super(MainDeep, self).__init__()
		'''
		Your code here
		'''
		self.fun1 = nn.Linear(2,100)
		self.relu = nn.ReLU()
		self.fun2 = nn.Linear(100,100)
		self.fun3 = nn.Linear(100,2)

	def forward(self, x):
		'''
		Your code here
		'''

		x = self.fun1(x)
		x = self.relu(x)
		x = self.fun2(x)
		x = self.relu(x)
		x = self.fun3(x)

		return x
