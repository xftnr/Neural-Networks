import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import load
from .models import *

dirname = os.path.dirname(os.path.abspath(__file__))

def test(iterations, batch_size=256):
	train_inputs, train_labels = load(os.path.join('tux_valid.dat'))
	
	model = ConvNetModel()
	pred = lambda x: np.argmax(x.detach().numpy(), axis=1)

	model.load_state_dict(torch.load(os.path.join(dirname, 'convnet.th')))
	model.eval()

	accuracies = []
	for iteration in range(iterations):
		# Construct a mini-batch
		batch = np.random.choice(train_inputs.shape[0], batch_size)
		batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32)
		batch_inputs = batch_inputs.permute(0, 3, 1, 2)
		pred_val = pred(model(batch_inputs.view(batch_size, 3, 64, 64)))
		accuracies.append( np.mean(pred_val == train_labels[batch]) )
	print( 'Accuracy ', np.mean(accuracies), '+-', np.std(accuracies)/np.sqrt(len(accuracies)))


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--iterations', type=int, default=10)
	args = parser.parse_args()

	print ('[I] Testing')
	test(args.iterations)
