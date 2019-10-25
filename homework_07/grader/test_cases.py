from . import base
import torch, os
from skimage import io
import numpy as np
from torchvision import transforms

from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

def one_hot(x, n=6):
	return x.view(-1,1) == torch.arange(6, dtype=x.dtype, device=x.device)[None]

def J(M):
	return (M.diag() / (M.sum(dim=0) + M.sum(dim=1) - M.diag() + 1e-5)).mean()

def CM(outputs, true_labels):
	return torch.matmul( one_hot(outputs).t().float(), one_hot(true_labels).float() )

def iou(outputs, true_labels):
	return J(CM(outputs, true_labels))

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
	INPUT_EXAMPLE = (torch.rand([32,3,256,256]),)

	def __init__(self, module):
		import torch
		self.module = module
		self.convnet = module.convnet
		self.g = torch.jit.get_trace_graph(self.convnet, self.INPUT_EXAMPLE)[0].graph()
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
		pred = lambda x: np.argmax(x.detach().numpy(), axis=1)
		M = getattr(self, 'convnet')
		M.eval()
		if M is None:
			print( 'Not implemented' )

		CMs = []
		for data in load('valid', num_workers=4, batch_size=32):
			images = data['inp_image']
			labels = data['lbl_image'].long()
			model_outputs = M(images)
			model_pred = torch.argmax(model_outputs, dim=1)
			
			CMs.append(CM(model_pred, labels))

		mean_iou = J(torch.stack(CMs).mean(dim=0))

		with self.SECTION('Testing validation IoU'):
			for k in np.linspace(0.2, 0.4, 100):
				self.CASE(mean_iou >= k)
		
