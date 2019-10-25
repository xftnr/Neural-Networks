"""
Module that contains that contains a couple of utility functions
"""

import numpy as np
import os

def load(filename, W=64, H=64):
	
	"""
	Loads the data that is provided
	@param filename: The name of the data file. Can be either 'tux_train.dat' or 'tux_val.dat'
	@return images: Numpy array of all images where the shape of each image will be W*H*3
	@return labels: Array of integer labels for each corresponding image in images
	"""

	try:
		data = np.fromfile(filename, dtype=np.uint8).reshape((-1, W*H*3+1))
	except Exception as e:
		print('Check if the filepath of the dataset is {}'.format(os.path(filename)))
		
	images, labels = data[:, :-1].reshape((-1,H,W,3)), data[:, -1]
	return images, labels

class SummaryWriter:
	def __init__(self, *args, **kwargs):
		print("tensorboardX not found. You need to install it to use the SummaryWriter.")
		print("try: pip3 install tensorboardX")
		raise ImportError

try:
	from tensorboardX import SummaryWriter
except ImportError:
	pass
