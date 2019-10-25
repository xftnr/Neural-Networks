"""
Module that contains that some utility functions
"""

import os
import numpy as np
import torch

from torch.utils.data import DataLoader, Dataset

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
		# Delete any short seqs
		if self.crop is not None:
			self.action_seqs = [ s for s in self.action_seqs if len(s) > self.crop ]

	def __len__(self):
		return len(self.action_seqs)
	
	def __getitem__(self, idx):
		r = self.action_seqs[idx]
		if self.crop is not None:
			s = np.random.choice(len(r) - self.crop + 1)
			r = r[s:s+self.crop]
		return np.unpackbits(r[None], axis=0)[:6]
		
def load(data_filepath, num_workers=0, batch_size=32, **kwargs):
	dataset = ActionDataset(data_filepath, **kwargs)
	return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

class SummaryWriter:
	def __init__(self, *args, **kwargs):
		print("tensorboardX not found. You need to install it to use the SummaryWriter.")
		print("try: pip3 install tensorboardX")
		raise ImportError
try:
	from tensorboardX import SummaryWriter
except ImportError:
	pass
