from . import base
import torch, os
import numpy as np

from torch.utils.data import DataLoader, Dataset
from torch import nn
from pytux import Tux


def get_action(logits):
	probs = 1. / (1. + np.exp(-logits.detach().numpy()))
	bits = np.array([np.random.uniform()<=p for p in probs]).astype(int)
	return int(np.packbits(bits)[0] >> 2)

class ActionDataset(Dataset):
	def __init__(self, data_dir, crop=None):
		self.data_dir = data_dir
		self.crop = crop
		
		self.trajs = os.listdir(data_dir)
		
		self._cache = {}
		
	
	def __len__(self):
		return len(self.trajs)//2
		
	def __getitem__(self, idx):
		if idx not in self._cache:
			imgs = np.load(os.path.join(self.data_dir, '%04d_img.npy'%idx))
			actions = np.load(os.path.join(self.data_dir, '%04d_action.npy'%idx))
			
			self._cache[idx] = (imgs, actions)
		
		imgs, actions = self._cache[idx]
		
		if self.crop is not None:
			s = np.random.choice(len(imgs) - self.crop + 1)
			imgs = imgs[s:s+self.crop]
			actions = actions[s:s+self.crop]
		
		imgs = (imgs - [4.417,3.339,4.250]) / [23.115,19.552,20.183]

		return imgs, np.unpackbits(actions[None], axis=0)[2:]

class Grader(base.Grader):
	TOTAL_SCORE = 100

	def __init__(self, module):
		import torch
		self.module = module
		self.model = module.model#.cuda()
		if self.model is None:
			print( "Failed to load model. Did you train one?" )
			exit(1)
		self.model.eval()
		self.verbose = False

		self.levels = [
			'01 - Welcome to Antarctica.stl',
			'02 - The Journey Begins.stl',
			'03 - Via Nostalgica.stl',
			'04 - Tobgle Road.stl',
			'05 - The Somewhat Smaller Bath.stl',
		]

	def op_check(self):
		pass
	
	def grade(self):
		import numpy as np
		
		torch.manual_seed(0)
		# Load the data
		data = ActionDataset('val')
		loss = nn.BCEWithLogitsLoss()
		
		loss_vals = []
		for i in range(len(data)):
			obs, actions = data[i]
			obs = torch.as_tensor(obs).float()#.cuda()
			actions = torch.as_tensor(actions).float()#.cuda()
			
			obs = obs[None].float().permute(0,1,4,2,3)#.cuda()
			actions = actions[None].float().permute(0,2,1)#.cuda()
			
			pred = self.model.policy()
			
			for i in range(obs.shape[1]):
				# Evaluate the prediction accuracy
				model_output = pred(obs[0,i,...])
				action = actions[0,i,...]
								
				l = float(loss(model_output, action))
				loss_vals.append(l)

		median_ll = np.median(loss_vals)

		positions = {level : [] for level in self.levels}

		
		for it in range(10):
			for level in self.levels:
				p = 0.0
				T = Tux('data/levels/world1/%s'%level, 128, 128, acting=True, visible=True, synchronized=True)

				# Restart Tux
				T.restart()
				if not T.waitRunning():
					exit(0)

				fid, act, state, obs = T.step(0)
				policy = self.model.policy()

				for t in range(2000):
					tux_mask = (obs['label'] % 8) == 4

					xs, ys = np.argwhere(tux_mask).T
					try:
						x = int(xs.mean())
						y = int(ys.mean())
					except:
						x = 64
						y = 64

					img = obs['image']
					img = np.pad(img, ((32,32), (32,32), (0,0)), mode='constant', constant_values=127)
					img = img[x:x+64,y:y+64]
					img = (img - [4.417,3.339,4.250]) / [23.115,19.552,20.183]
					img = torch.as_tensor(img).float()
					logits = policy(img.permute(2,0,1))
					a = get_action(logits)
					try:
						fid, act, state, obs = T.step(a)
					except TypeError as e:
						print (e)
						break
					if state['is_dying']:
						break
					p = max(p, state['position'])
				positions[level].append(p)
		
	
		with self.SECTION('Testing validation Loss'):
			for k in np.linspace(0.1, 0.5, 50):
				self.CASE(median_ll <= k)
				
		with self.SECTION('Testing Level 1 performance'):
			for k in np.linspace(0.1, 0.24, 10):
				self.CASE(np.mean(positions['01 - Welcome to Antarctica.stl']) >= k)

		with self.SECTION('Testing Level 2 performance'):
			for k in np.linspace(0.03, 0.18, 10):
				self.CASE(np.mean(positions['02 - The Journey Begins.stl']) >= k)

		with self.SECTION('Testing Level 3 performance'):
			for k in np.linspace(0.01, 0.16, 10):
				self.CASE(np.mean(positions['03 - Via Nostalgica.stl']) >= k)

		with self.SECTION('Testing Level 4 performance'):
			for k in np.linspace(0.04, 0.14, 10):
				self.CASE(np.mean(positions['04 - Tobgle Road.stl']) >= k)
		
		with self.SECTION('Testing Level 5 performance'):
			for k in np.linspace(0.05, 0.1, 10):
				self.CASE(np.mean(positions['05 - The Somewhat Smaller Bath.stl']) >= k)