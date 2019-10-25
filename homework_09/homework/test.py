import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import *
from .models import *

dirname = os.path.dirname(os.path.abspath(__file__))

scale_factor = 1.54

def test(iterations):
	# Load the model
	model = SeqModel()#.cuda()
	model.load_state_dict(torch.load(os.path.join(dirname, 'model.th')))
	model.eval()
	
	# Load the data
	data = ActionDataset('valid.dat')
	loss = nn.BCEWithLogitsLoss(reduction='sum')
	
	loss_vals = []
	for i in range(iterations):
		seq = torch.as_tensor(data[i]).float()
		pred = model.predictor()
		
		prob = None
		for i in range(seq.shape[-1]):
			if prob is not None:
				# Evaluate the prediction accuracy
				loss_vals.append( float(loss(prob, seq[:,i])) )
			prob = pred(seq[:,i])
	print('Mean log-likelihood loss', np.mean(loss_vals))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--iterations', type=int, default=32)
	args = parser.parse_args()

	print ('[I] Testing')
	test(args.iterations)
