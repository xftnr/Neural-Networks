import argparse, pickle
import numpy as np
import os
from itertools import cycle

import torch
from torch import nn, optim
from torchvision import transforms

# Use the util.load() function to load your dataset
from .utils import load
from .models import *

dirname = os.path.dirname(os.path.abspath(__file__))

def one_hot(x, n=6):
	return x.view(-1,1) == torch.arange(6, dtype=x.dtype, device=x.device)[None]

def J(M):
	return (M.diag() / (M.sum(dim=0) + M.sum(dim=1) - M.diag() + 1e-5)).mean()

def CM(outputs, true_labels):
	return torch.matmul( one_hot(outputs).t().float(), one_hot(true_labels).float() )

def iou(outputs, true_labels):
	return J(CM(outputs, true_labels))

train_class_loss_weights = np.array([

    # Your code here
	155/1000, 1700/1000, 1400/1000, 1700/1000, 1700/1000, 420/1000

])


def train(max_iter, batch_size=64, log_dir=None):
	'''
	This is the main training function, feel free to modify some of the code, but it
	should not be required to complete the assignment.
	'''

	"""
	Load the training data
	"""

	train_dataloader = load('train', num_workers=4, crop=64)
	valid_dataloader = load('valid', num_workers=4)
	train_dataloader_iterator = cycle(iter(train_dataloader))
	valid_dataloader_iterator = cycle(iter(valid_dataloader))

	model = FConvNetModel()

	log = None
	if log_dir is not None:
		from .utils import SummaryWriter
		log = SummaryWriter(log_dir)

	# If your model does not train well, you may swap out the optimizer or change the lr

	#optimizer = optim.Adam(model.parameters(), lr = 1e-4, weight_decay=5e-4)
	optimizer = optim.SGD(model.parameters(), lr = 5e-3, momentum=0.9, weight_decay=1e-4)

	loss = nn.CrossEntropyLoss(weight=torch.from_numpy(train_class_loss_weights).float())

	for t in range(max_iter):
		train_batch = next(train_dataloader_iterator)
		batch_inputs = train_batch['inp_image']
		batch_labels = train_batch['lbl_image'].long()

		model.train()

		# zero the gradients (part of pytorch backprop)
		optimizer.zero_grad()

		# Compute the model output and loss (view flattens the input)
		model_outputs = model(batch_inputs)
		model_pred = torch.argmax(model_outputs, dim=1)
		t_loss_val = loss(model_outputs, batch_labels)
		t_acc_val = iou(model_pred, batch_labels)

		# Compute the gradient
		t_loss_val.backward()

		# Update the weights
		optimizer.step()

		if t % 10 == 0:
			model.eval()

			valid_batch = next(valid_dataloader_iterator)
			batch_inputs = valid_batch['inp_image']
			batch_labels = valid_batch['lbl_image'].long()

			model_outputs = model(batch_inputs)
			model_pred = torch.argmax(model_outputs, dim=1)
			v_acc_val = iou(model_pred, batch_labels)

			print('[%5d]'%t, 'loss = %f'%t_loss_val, 't_iou = %f'%t_acc_val, 'v_iou = %f'%v_acc_val)

			if log is not None:
				log.add_scalar('train/loss', t_loss_val, t)
				log.add_scalar('train/iou', t_acc_val, t)
				log.add_scalar('val/iou', v_acc_val, t)

	# Save the trained model
	torch.save(model.state_dict(), os.path.join(dirname, 'fconvnet.th')) # Do NOT modify this line

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--max_iter', type=int, default=10000)
	parser.add_argument('-l', '--log_dir')
	args = parser.parse_args()

	print ('[I] Start training')
	train(args.max_iter, log_dir=args.log_dir)
	print ('[I] Training finished')
