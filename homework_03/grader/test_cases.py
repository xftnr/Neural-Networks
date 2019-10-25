from . import base
import torch, os

target_accuracies = {
'regress' : 0.20,
'one_hot' : 0.75,
'll' : 0.8,
'l2' : 0.8
}
expected_output = {
'RegressLoss': [4.809960842132568,
8.849384307861328,
8.747788429260254,
10.090002059936523,
4.776233196258545],
'OnehotLoss':[7.138614654541016,
7.098428726196289,
6.970293998718262,
5.6532487869262695,
6.733926773071289],
'LlLoss':[1.9842205047607422,
2.287382125854492,
2.421297311782837,
1.8271944522857666,
1.8931397199630737],
'L2Loss': [0.9268025159835815,
0.9895575046539307,
1.0157736539840698,
0.9065024256706238,
0.930397629737854]
}
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
	INPUT_EXAMPLE = (torch.rand([32,64*64*3]),)

	def __init__(self, module):
		import torch
		self.module = module
		self.regress = module.regress
		self.one_hot = module.one_hot
		self.ll = module.ll
		self.l2 = module.l2
		self.verbose = False
	
	def op_check(self):
		pass
	
	def grade(self):
		import numpy as np
		
		test_inputs, test_labels = load(os.path.join('tux_valid.dat'))
		
		torch.manual_seed(0)
		pred = lambda x: x.detach().numpy().round().astype(int)
		for m in ['regress', 'one_hot', 'll', 'l2']:
			with self.SECTION("Validation accuracy for '%s' within range %0.2f"%(m, target_accuracies[m])):
				M = getattr(self, m)
				if M is None:
					print( 'Not implemented' )
				else:
					accuracies = []
					for i in range(0, len(test_inputs)-256, 256):
						batch_inputs = torch.as_tensor(test_inputs[i:i+256], dtype=torch.float32)
						pred_val = pred(M(batch_inputs.view(-1, 64*64*3)))
						accuracies.extend( pred_val == test_labels[i:i+256] )
					acc = np.mean(accuracies)
					for k in np.linspace(target_accuracies[m]/2, target_accuracies[m], 20):
						self.CASE(acc >= k)
			
			# Make sure we use one-hot predictions for the rest
			pred = lambda x: np.argmax(x.detach().numpy(), axis=1)
		
		for m in ['RegressLoss', 'OnehotLoss', 'LlLoss', 'L2Loss']:
			with self.SECTION("Testing '%s'"%m):
				L = getattr(self.module, m)()
				for it in range(5):
					torch.manual_seed(it)
				
					dummy_label = torch.randint(0, 6, (16,), dtype=torch.long)
					if m == 'RegressLoss':
						dummy_input = torch.rand(16)*6 - 0.5
						dummy_label = dummy_label.float()
					else:
						dummy_input = torch.randn(16, 6).float()
					self.CASE( abs(expected_output[m][it] - float(L(dummy_input, dummy_label))) < 0.2 )
