from . import base
import torch, os
from torchvision import transforms

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
	INPUT_EXAMPLE = (torch.rand([32,3,64,64]),)

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
		
		test_inputs, test_labels = load(os.path.join('tux_valid.dat'))

		torch.manual_seed(0)
		pred = lambda x: np.argmax(x.detach().numpy(), axis=1)
		M = getattr(self, 'convnet')
		M.eval()
		if M is None:
			print( 'Not implemented' )
		nodes = list(self.g.nodes())
		init_num = len(nodes)
		last_num = 0
		last_op = 0
		for op in self.g.nodes():
			num = self.get_op_num(op)
			if num > last_num:
				last_num = num
				last_op = op
			if num < init_num:
				init_num = num
		
		conv_depth = self.get_layer_depth(last_op, '_convolution', init_num, 0)
		if conv_depth < 15:
			print ("Current model depth %d, below 20. Please change your model definition" % conv_depth)
		else:
			with self.SECTION('Testing validation accuracy'):
				accuracies = []
				for i in range(0, len(test_inputs)-256, 256):
					batch_inputs = transform_val(test_inputs[i:i+256])
					pred_val = pred(M(batch_inputs))
					accuracies.extend( pred_val == test_labels[i:i+256] )
				acc = np.mean(accuracies)
				for k in np.linspace(0.94, 0.96, 100):
					self.CASE(acc >= k)