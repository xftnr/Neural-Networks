from . import base
import torch, os
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch import nn

class ActionDataset(Dataset):

	"""
	Dataset class that reads the Actions dataset
	"""

	def __init__(self, data_filepath, crop=None):
				
		self.crop = crop

		# Load all data into memory
		print ("[I] Loading data from %s"%data_filepath)
		with open(data_filepath, 'r') as f:
			self.action_seqs = [ np.fromstring(l, dtype=np.uint8, sep=',') for l in f.readlines() ]

	def __len__(self):
		return len(self.action_seqs)
	
	def __getitem__(self, idx):
		r = self.action_seqs[idx]
		return np.unpackbits(r[None], axis=0)[:6]

class Grader(base.Grader):
	TOTAL_SCORE = 100
	INPUT_EXAMPLE = torch.randint(0, 2, (32, 100))

	def __init__(self, module):
		import torch
		self.module = module
		self.model = module.model
		if self.model is None:
			print( "Failed to load model. Did you train one?" )
			exit(1)
		self.model.eval()
		self.verbose = False

	def op_check(self):
		pass
	
	def grade(self):
		import numpy as np
		
		torch.manual_seed(0)
		# Load the data
		data = ActionDataset('valid.dat')
		loss = nn.BCEWithLogitsLoss(reduction='sum')
		
		loss_vals = []
		for k in range(len(data)):
			seq = torch.as_tensor(data[k]).float()
			pred = self.model.predictor()
			
			prob = None
			for i in range(seq.shape[-1]):
				if prob is not None:
					# Evaluate the prediction accuracy
					loss_vals.append( float(loss(prob, seq[:,i])) )
				prob = pred(seq[:,i])
		
		mean_ll = np.mean(loss_vals)
		print( mean_ll )
		with self.SECTION('Testing validation Loss'):
			for k in np.linspace(0.1, 0.2, 100):
				self.CASE(mean_ll <= k)
