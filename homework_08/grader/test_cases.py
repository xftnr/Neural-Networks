from . import base
import torch, os
from skimage import io
import numpy as np
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torch import nn

downsample = nn.AvgPool2d(5, 4, 2)

def get_rgb(batch_images):

	channel_1 = (batch_images[:, 0, :, :] * 0.246 + 0.354)*255
	channel_2 = (batch_images[:, 1, :, :] * 0.286 + 0.488)*255
	channel_3 = (batch_images[:, 2, :, :] * 0.362 + 0.564)*255

	return torch.clamp(torch.stack((channel_1, channel_2, channel_3), 1).long().float(), 0, 255)

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
		return {
			'inp_image' : self.image_transform(I),
			'lbl_image' : L
		}

def load(data_folder, num_workers=0, batch_size=32, **kwargs):
	dataset = TuxDataset(data_folder, **kwargs)
	return DataLoader(dataset, batch_size=batch_size, num_workers=num_workers, shuffle=True)

class Grader(base.Grader):
	TOTAL_SCORE = 100
	INPUT_EXAMPLE = (torch.rand([32,3,64,64]), torch.rand([32,256,256]))

	def __init__(self, module):
		import torch
		self.module = module
		self.convnet = module.convnet
		self.convnet.eval()
		self.verbose = False
		
	def get_op_num(self, op):
		num = int(str(op).split(':')[0].split('%')[1])
		return num
		
	def get_layer_depth(self, op, layer_name, init_num, depth):
		# If all input of op are input nodes, reached top and returns
		if all([self.get_op_num(i.node()) < init_num for i in op.inputs()]):
			return depth
		
		k = op.kind()
		if 'aten::' in k:
			k = k.replace('aten::','')
		if k == layer_name:
			depth += 1
		
		# Return the maximum of all depths of its input nodes 
		return max([self.get_layer_depth(i.node(), layer_name, init_num, depth) for i in op.inputs()])
				
	def op_check(self):
		pass
	
	def grade(self):
		import numpy as np
		
		torch.manual_seed(0)
		loss = nn.L1Loss()
		M = getattr(self, 'convnet')
		M.eval()
		if M is None:
			print( 'Not implemented' )

		l1_loss = []
		for data in load('valid', num_workers=4, batch_size=32):
			targets = data['inp_image']
			labels = data['lbl_image'].long()
			images = downsample(targets)

			model_outputs = M(images, labels)
			model_pred = get_rgb(model_outputs)
			
			l1_loss.append(loss(model_pred, get_rgb(targets)))

		mean_l1 = np.mean(l1_loss)
		print( mean_l1 )
		with self.SECTION('Testing validation L1 Loss'):
			for k in np.linspace(14, 9, 100):
				self.CASE(mean_l1 <= k)
		
		with self.SECTION('Testing Extra Credit'):
			for k in [8,7,6,5]:
				for _ in range(5):
					self.CASE(mean_l1 <= k)
