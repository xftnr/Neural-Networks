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

downsample = nn.AvgPool2d(5, 4, 2)

def get_rgb(batch_images):

	channel_1 = (batch_images[:, 0, :, :] * 0.246 + 0.354)*255
	channel_2 = (batch_images[:, 1, :, :] * 0.286 + 0.488)*255
	channel_3 = (batch_images[:, 2, :, :] * 0.362 + 0.564)*255

	return torch.clamp(torch.stack((channel_1, channel_2, channel_3), 1).long().float(), 0, 255)

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
	print ("Num of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

	log = None
	if log_dir is not None:
		from .utils import SummaryWriter
		log = SummaryWriter(log_dir)

	# If your model does not train well, you may swap out the optimizer or change the lr

	optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=5e-4)

	loss = nn.L1Loss(reduce=True)

	for t in range(max_iter):
		train_batch = next(train_dataloader_iterator)
		batch_targets = train_batch['inp_image']
		batch_labels = train_batch['lbl_image'].long()

		# .detach() is used to avoid backpropagation to the original high resolution image
		batch_inputs = downsample(batch_targets).detach()
		model.train()

		# zero the gradients (part of pytorch backprop)
		optimizer.zero_grad()

		# Compute the model output and loss (view flattens the input)
		model_outputs = model(batch_inputs, batch_labels)
		t_loss_val = loss(model_outputs, batch_targets)

		t_rgb_loss_val = loss(get_rgb(model_outputs), get_rgb(batch_targets))

		# Compute the gradient
		t_loss_val.backward()

		# Update the weights
		optimizer.step()

		if t % 10 == 0:
			model.eval()

			valid_batch = next(valid_dataloader_iterator)
			batch_targets = valid_batch['inp_image']
			batch_labels = valid_batch['lbl_image'].long()
			batch_inputs = downsample(batch_targets).detach()

			model_outputs = model(batch_inputs, batch_labels)
			v_rgb_loss_val = loss(get_rgb(model_outputs), get_rgb(batch_targets))

			print('[%5d]'%t, 't_loss = %f'%t_loss_val, 't_rgb_loss = %f'%t_rgb_loss_val, 'v_rgb_loss = %f'%v_rgb_loss_val)

			if log is not None:
				log.add_scalar('train/loss', t_loss_val, t)
				log.add_scalar('val/loss', v_loss_val, t)

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
