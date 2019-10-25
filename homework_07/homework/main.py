from .models import *
import os, torch

dirname = os.path.dirname(os.path.abspath(__file__))

try:
	convnet = FConvNetModel()
	convnet.load_state_dict(torch.load(os.path.join(dirname, 'fconvnet.th')))
except:
	convnet = None
