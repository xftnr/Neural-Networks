import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import load
from .models import *

dirname = os.path.dirname(os.path.abspath(__file__))

def test(model_name, iterations, batch_size=256):
	train_inputs, train_labels = load(os.path.join('tux_valid.dat'))
	
	if model_name == 'regress':
		model = ScalarModel()
		pred = lambda x: x.detach().numpy().round().astype(int)
	else:
		model = VectorModel()
		pred = lambda x: np.argmax(x.detach().numpy(), axis=1)
	
	model.load_state_dict(torch.load(os.path.join(dirname, model_name + '.th')))
	
	accuracies = []
	for iteration in range(iterations):
		# Construct a mini-batch
		batch = np.random.choice(train_inputs.shape[0], batch_size)
		batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32)
		
		pred_val = pred(model(batch_inputs.view(batch_size, -1)))
		accuracies.append( np.mean(pred_val == train_labels[batch]) )
	print( 'Accuracy ', np.mean(accuracies), '+-', np.std(accuracies) )


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('model', choices=['regress', 'oneHot', 'LL', 'L2'])
	parser.add_argument('-i', '--iterations', type=int, default=10)
	args = parser.parse_args()

	print ('[I] Testing %s'%args.model)
	test(args.model, args.iterations)
