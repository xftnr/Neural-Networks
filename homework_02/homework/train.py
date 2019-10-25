import argparse, pickle, os
import torch.nn as nn
from torch import tensor, save
import torch.optim as optim
import numpy as np

import torch
from .main import MainLinear, MainDeep


def train_linear(model):
	'''
	Your code here
	'''
	model = MainLinear()
	criterion = nn.MSELoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	epochs = 10
	# running_loss = 0.0
	targets = torch.zeros([10000,2], dtype = torch.float32)
	inputs = torch.rand([10000,2], dtype=torch.float32)

	count = 0
	for i in inputs:
		if i[0]**2 + i[1]**2 <= 1:
			targets[count] = torch.tensor([1,0], dtype = torch.float32)
		else:
			targets[count] = torch.tensor([-1,0], dtype = torch.float32)
		count += 1

	# targets = torch.trunc(targets)

	running_loss = 0.0
	for ep in range(epochs):

		for i in range(0,100):
		# targets = torch.rand([10000,1], dtype=torch.float32))

			out = model(inputs)
			loss = criterion(out, targets)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		running_loss = 0.


	# Save the trained model
	dirname = os.path.dirname(os.path.abspath(__file__)) # Do NOT modify this line
	save(model.state_dict(), os.path.join(dirname, 'linear')) # Do NOT modify this line


def train_deep(model):
	'''
	Your code here
	'''
	model = MainDeep()
	criterion = nn.BCEWithLogitsLoss()
	optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
	epochs = 50
	# running_loss = 0.0
	targets = torch.zeros([10000,2], dtype = torch.float32)
	inputs = torch.rand([10000,2], dtype=torch.float32)

	count = 0
	for i in inputs:
		if i[0]**2 + i[1]**2 <= 1:
			targets[count] = torch.tensor([1,0], dtype = torch.float32)
		else:
			targets[count] = torch.tensor([0,1], dtype = torch.float32)
		count += 1

	# targets = torch.trunc(targets)

	running_loss = 0.0
	for ep in range(epochs):

		for i in range(0,100):
		# targets = torch.rand([10000,1], dtype=torch.float32))

			out = model(inputs)
			loss = criterion(out, targets)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
			running_loss += loss.item()
		running_loss = 0.

	# Save the trained model
	dirname = os.path.dirname(os.path.abspath(__file__)) # Do NOT modify this line
	save(model.state_dict(), os.path.join(dirname, 'deep')) # Do NOT modify this line


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('model', choices=['linear', 'deep'])
	args = parser.parse_args()

	if args.model == 'linear':
		print ('[I] Start training linear model')
		train_linear(MainLinear())
	elif args.model == 'deep':
		print ('[I] Start training linear model')
		train_deep(MainDeep())

	print ('[I] Training finished')
