from torch import nn
import torch.nn.functional as F
import torch
import numpy as np

class Policy:
	def __init__(self, model):
		'''
		Your code here
		'''
		self.model = model
		
	def __call__(self, obs):
		'''
		Your code here
		'''
		return obs

class Model(nn.Module):
	def __init__(self):
		super().__init__()
		'''
		Your code here
		'''
		
	def forward(self, hist):
		'''
		Your code here
		'''
		pass

	def policy(self):
		return Policy(self)
