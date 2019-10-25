from .models import *
import os, torch

dirname = os.path.dirname(os.path.abspath(__file__))

try:
	convnet = ConvNetModel()
	convnet.load_state_dict(torch.load(os.path.join(dirname, 'convnet.th')))
except:
	convnet = None