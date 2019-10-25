from .models import *
import os, torch

dirname = os.path.dirname(os.path.abspath(__file__))

try:
	regress = ScalarModel()
	regress.load_state_dict(torch.load(os.path.join(dirname, 'regress.th')))
except:
	regress = None

try:
	one_hot = VectorModel()
	one_hot.load_state_dict(torch.load(os.path.join(dirname, 'oneHot.th')))
except:
	one_hot = None

try:
	ll = VectorModel()
	ll.load_state_dict(torch.load(os.path.join(dirname, 'LL.th')))
except:
	ll = None

try:
	l2 = VectorModel()
	l2.load_state_dict(torch.load(os.path.join(dirname, 'L2.th')))
except:
	l2 = None

from .train import RegressLoss, OnehotLoss, LlLoss, L2Loss