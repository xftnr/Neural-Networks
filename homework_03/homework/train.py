import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import load
from .models import *

dirname = os.path.dirname(os.path.abspath(__file__))

class RegressLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, model_output, label):
		"""
		model_output: size (N, 1)
		label:  size (N, 1)
		return value: scalar
		Your code here
		"""
		loss = torch.mean((model_output-label)**2)
		return loss
		# parameters
		# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
		# optimizer.zero_grad()
		# loss.backward()
		# optimizer.step()

def label2onehot(label):
	# Transform a label of size (N,1) into a one-hot encoding of size (N,6)
	return (label[:,None] == torch.arange(6)[None]).float()

class OnehotLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, model_output, label):
		"""
		model_output: size (N, 6)
		label:  size (N, 1)
		return value: scalar
		Your code here
		"""
		onehotlabel = label2onehot(label)
		loss = (torch.norm(model_output-onehotlabel)**2)/len(model_output)
		return loss
		# may be need a for loop
		# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
		# optimizer.zero_grad()
		# loss.backward()
		# optimizer.step()

class LlLoss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, model_output, label):
		"""
		model_output: size (N, 6)
		label:  size (N, 1)
		return value: scalar
		Your code here
		"""
		# onehotlabel = torch.tensor(label2onehot(label), dtype = torch.long)
		criterion = nn.CrossEntropyLoss()
		loss = criterion(model_output,label)
		return loss
		# may be need a for loop
		# optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
		# optimizer.zero_grad()
		# loss.backward()
		# optimizer.step()

class L2Loss(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, model_output, label):
		"""
		model_output: size (N, 6)
		label:  size (N, 1)
		return value: scalar
		Your code here
		"""
		criterion = nn.Softmax(dim = 1)
		soft = criterion(model_output)
		onehotlabel = label2onehot(label)
		loss = (torch.norm(soft-onehotlabel)**2)/len(model_output)
		return loss

def train(model_name, iterations, batch_size=64):
	'''
	This is the main training function, feel free to modify some of the code, but it
	should not be required to complete the assignment.
	'''

	"""
	Load the training data
	"""

	train_inputs, train_labels = load(os.path.join('tux_train.dat'))

	loss = eval(model_name.capitalize()+'Loss')()
	if model_name == 'regress':
		model = ScalarModel()
	else:
		model = VectorModel()

	# We use the ADAM optimizer with default learning rate (lr)
	# If your model does not train well, you may swap out the optimizer or change the lr
	optimizer = optim.Adam(model.parameters(), lr = 1e-4)

	for iteration in range(iterations):
		# Construct a mini-batch
		batch = np.random.choice(train_inputs.shape[0], batch_size)
		batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32)
		batch_labels = torch.as_tensor(train_labels[batch], dtype=torch.long)

		if model_name == 'regress': # Regression expects float labels
			batch_labels = batch_labels.float()

		# zero the gradients (part of pytorch backprop)
		optimizer.zero_grad()

		# Compute the model output and loss (view flattens the input)
		loss_val = loss( model(batch_inputs.view(batch_size, -1)), batch_labels)

		# Compute the gradient
		loss_val.backward()

		# Update the weights
		optimizer.step()

		if iteration % 10 == 0:
			print('[%5d]'%iteration, 'loss = %f'%loss_val)

	# Save the trained model
	torch.save(model.state_dict(), os.path.join(dirname, model_name + '.th')) # Do NOT modify this line

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('model', choices=['regress', 'oneHot', 'LL', 'L2'])
	parser.add_argument('-i', '--iterations', type=int, default=10000)
	args = parser.parse_args()

	print ('[I] Start training %s'%args.model)
	train(args.model, args.iterations)
	print ('[I] Training finished')
