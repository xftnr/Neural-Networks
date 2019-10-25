import argparse, pickle
import numpy as np
import os
from itertools import cycle

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import *
from .models import *

dirname = os.path.dirname(os.path.abspath(__file__))

def cycle(seq):
	while True:
		for elem in seq:
			yield elem

def train(max_iter, batch_size=64, log_dir=None):
	'''
	This is the main training function, feel free to modify some of the code, but it
	should not be required to complete the assignment.
	'''

	"""
	Load the training data
	"""

	train_dataloader = load('train.dat', num_workers=4, crop=100)
	valid_dataloader = load('valid.dat', num_workers=4, crop=100)

	train_dataloader_iterator = cycle(train_dataloader)
	valid_dataloader_iterator = cycle(valid_dataloader)

	model = SeqModel()

	print ("Num of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

	log = None
	if log_dir is not None:
		from .utils import SummaryWriter
		log = SummaryWriter(log_dir)
	
	# If your model does not train well, you may swap out the optimizer or change the lr
	
	optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=1e-4)
	#optimizer = optim.SGD(model.parameters(), lr = 1e-2, momentum=0.9, weight_decay=1e-4)
	
	loss = nn.BCEWithLogitsLoss()
	
	for t in range(max_iter):
		batch = next(train_dataloader_iterator).float()
		batch_inputs = batch[:,:,:-1]
		batch_outputs = batch[:,:,1:]

		model.train()
		
		# zero the gradients (part of pytorch backprop)
		optimizer.zero_grad()
		
		# Compute the model output and loss (view flattens the input)
		model_outputs, _ = model(batch_inputs, None)

		# Compute the loss
		t_loss_val = loss(model_outputs, batch_outputs)*6
		
		# Compute the gradient
		t_loss_val.backward()
	
		# Update the weights
		optimizer.step()
	
		if t % 10 == 0:
			model.eval()
			
			valid_batch = next(valid_dataloader_iterator).float()
			batch_inputs = batch[:,:,:-1]
			batch_outputs = batch[:,:,1:]
			
			model_outputs, _ = model(batch_inputs, None)
			
			v_loss_val = loss(model_outputs, batch_outputs)*6

			print('[%5d]  t_loss = %f   v_loss_val = %f'%(t, t_loss_val,v_loss_val))
			
			if log is not None:
				log.add_scalar('train/loss', t_loss_val, t)
				log.add_scalar('val/loss', v_loss_val, t)

	# Save the trained model
	torch.save(model.state_dict(), os.path.join(dirname, 'model.th')) # Do NOT modify this line

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--max_iter', type=int, default=2000)
	parser.add_argument('-l', '--log_dir')
	args = parser.parse_args()

	print ('[I] Start training')
	train(args.max_iter, log_dir=args.log_dir)
	print ('[I] Training finished')
