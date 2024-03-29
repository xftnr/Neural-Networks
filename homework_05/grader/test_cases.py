from . import base
import torch, os
from torchvision import transforms

# Stores the increments for crossing a particular threshold
EXTRA_CREDIT = [
	(0.945, 10),
	(0.95, 10),
	(0.955, 10)
]

def load(filename, W=64, H=64):
	import numpy as np
	try:
		data = np.fromfile(filename, dtype=np.uint8).reshape((-1, W*H*3+1))
	except Exception as e:
		print('Check if the filepath of the dataset is {}'.format(os.path(filename)))
		
	images, labels = data[:, :-1].reshape((-1,H,W,3)), data[:, -1]
	return images, labels

def transform_val(val_inputs):

    """
    During Evaluation we don't use data augmentation, we only Normalize the images
    """

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.429, 0.505, 0.517], std=[0.274, 0.283, 0.347])
        ])


    transformed_data = []
    for inp_i in val_inputs:
        transformed_data.append(transform(inp_i))
    
    transformed_data = torch.stack(transformed_data)

    return transformed_data

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
			accuracies = []
			for i in range(0, len(test_inputs)-256, 256):
				batch_inputs = transform_val(test_inputs[i:i+256])
				pred_val = pred(M(batch_inputs))
				accuracies.extend( pred_val == test_labels[i:i+256] )
			acc = np.mean(accuracies)
			print(acc)
			with self.SECTION('Testing validation accuracy'):
				for k in np.linspace(0.92, 0.94, 100):
					self.CASE(acc >= k)
			with self.SECTION('Extra Credit [validation]'):
				for threshold, increment in EXTRA_CREDIT:
					self.CASE(acc >= threshold, increment)
