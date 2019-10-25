from . import base
import torch, os

def load(filename, W=64, H=64):
	import numpy as np
	try:
		data = np.fromfile(filename, dtype=np.uint8).reshape((-1, W*H*3+1))
	except Exception as e:
		print('Check if the filepath of the dataset is {}'.format(os.path(filename)))
		
	images, labels = data[:, :-1].reshape((-1,H,W,3)), data[:, -1]
	return images, labels

class Grader(base.Grader):
	TOTAL_SCORE = 100
	INPUT_EXAMPLE = (torch.rand([32,3,3,64]),)

	def __init__(self, module):
		import torch
		self.module = module
		self.convnet = module.convnet
		self.verbose = False

	def op_check(self):
		pass
	
	def grade(self):
		import numpy as np
		
		test_inputs, test_labels = load(os.path.join('tux_valid.dat'))

		torch.manual_seed(0)
		pred = lambda x: np.argmax(x.detach().numpy(), axis=1)
		M = getattr(self, 'convnet')
		M.eval()
		if M is None:
			print( 'Not implemented' )
		else:
			with self.SECTION('Testing validation accuracy'):
				accuracies = []
				for i in range(0, len(test_inputs)-256, 256):
					batch_inputs = torch.as_tensor(test_inputs[i:i+256], dtype=torch.float32)
					batch_inputs = batch_inputs.permute(0, 3, 1, 2)
					pred_val = pred(M(batch_inputs))
					accuracies.extend( pred_val == test_labels[i:i+256] )
				acc = np.mean(accuracies)
				for k in np.linspace(0.8, 0.9, 100):
					self.CASE(acc >= k)
