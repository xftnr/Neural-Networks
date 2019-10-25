"""
Module that contains that some utility functions
"""

import os
import numpy as np
from skimage import io

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class TuxDataset(Dataset):

	"""
	Dataset class that reads the Tux dataset
	"""

	def __init__(self, data_folder, crop=None):
		from os import path
		from glob import glob
		
		self.data_folder = data_folder
		
		# Load all data into memory
		print ("[I] Loading data from %s"%data_folder)
		self.filenames = glob(path.join(data_folder, '*-img.png'))
		self.crop = crop
		self.image_transform = transforms.Compose([
			transforms.ToTensor(), 
			transforms.Normalize(mean=[0.354,0.488,0.564], std=[0.246,0.286,0.362]),
		])
	
	def __len__(self):
		return len(self.filenames)
	
	def _mask(self, im):
		r = (im[:,:,0] > 0).astype(np.uint8) + 2*(im[:,:,1] > 0).astype(np.uint8) + 4*(im[:,:,2] > 0).astype(np.uint8)
		r[r > 5] = 5
		return r
	
	def __getitem__(self, idx):
		I, L = io.imread(self.filenames[idx]), self._mask(io.imread(self.filenames[idx].replace('-img', '-lbl')))
		if self.crop is not None:
			y, x = np.random.choice(I.shape[0]-self.crop), np.random.choice(I.shape[1]-self.crop)
			I,L = I[y:self.crop+y, x:self.crop+x],L[y:self.crop+y, x:self.crop+x]
			if np.random.random() > 0.5:
				I, L = np.ascontiguousarray(I[:,::-1]), np.ascontiguousarray(L[:,::-1])
		
		return {
			'inp_image': self.image_transform(I),
			'lbl_image' : L
		}

def load(data_folder, num_workers=0, batch_size=32, **kwargs):
	dataset = TuxDataset(data_folder, **kwargs)
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
