from . import base
import torch

class GraderLinear(base.Grader):
	TOTAL_SCORE = 15
	BLACK_LIST = ['acos', 'asin', 'atan', 'atan2', 'sqrt', 'mean', 'sum', 'ge', 'le', 'eq', 'ne', 'gt', 'lt']
	INPUT_EXAMPLE = (torch.rand([10,2]),)

	def io_check(self):
		# Make sure input accepts a list of random numbers
		o = self.m( *self.INPUT_EXAMPLE )
		self.CHECK_SHAPE(o, [10,2], name='output')

	def grade(self):
		import numpy as np
		import os
		torch.manual_seed(0)

		dirname = os.path.dirname(os.path.abspath(__file__))
		try:
			checkpoint = torch.load(os.path.join(dirname, '../homework/linear'))
			self.m.load_state_dict(checkpoint)
		except FileNotFoundError:
			print ("[E] Could not found checkpoint, please make sure you train your linear model first")
			return 

		with self.SECTION("Testing linearity of linear classifier."):
			for it in range(1):
				with self.GROUP():
					# Find intersections
					# x
					start = -10.0
					end = 10.0
					while end - start > 0.01:
						point = (end + start) / 2.0
						inputs = torch.tensor([[point, 0]])
						out = self.m(inputs).detach().numpy()[0,0]
						if out > 0:
							start = point
						else:
							end = point

					xpoint = point
					# y
					start = -10.0
					end = 10.0
					while end - start > 0.01:
						point = (end + start) / 2.0
						inputs = torch.tensor([[0, point]])
						out = self.m(inputs).detach().numpy()[0,0]
						if out > 0:
							start = point
						else:
							end = point
					ypoint = point
					# Get slope
					slope = -xpoint / ypoint
					# Try some points
					xs = [0.1, 0.3, 0.5, 0.7, 0.9]
					eps = 0.1
					for x in xs:
						above = self.m(torch.tensor([[x, slope * x + eps]]))[0,0]
						below = self.m(torch.tensor([[x, slope * x - eps]]))[0,0]
						self.CASE(above != below)


		with self.SECTION("Testing classification accuracy of linear classifier with 10000 samples"):
			for threshold in np.arange(0.90, 0.95, 0.01):
				for it in range(1):
					with self.GROUP():
						inputs = np.random.uniform(size=[10000, 2])
						labels = (np.sum(inputs * inputs, axis=1) > 1.0).astype('float')
						linear_out = self.m(torch.tensor(inputs).float()).detach().numpy()
						linear_out = np.argmax(linear_out, axis=1)
						eq = (labels == linear_out).astype('float')
						accuracy = np.mean(eq)
						self.CASE(accuracy > threshold)
						self.CASE(accuracy < 1.0)

class GraderDeep(base.Grader):
	TOTAL_SCORE = 8
	BLACK_LIST = ['acos', 'asin', 'atan', 'atan2', 'sqrt', 'mean', 'sum']
	INPUT_EXAMPLE = (torch.rand([10,2]),)

	def io_check(self):
		# Make sure input accepts a list of random numbers
		o = self.m( *self.INPUT_EXAMPLE )
		self.CHECK_SHAPE(o, [10,2], name='output')

	def grade(self):
		import numpy as np
		import os
		torch.manual_seed(0)

		dirname = os.path.dirname(os.path.abspath(__file__))
		try:
			checkpoint = torch.load(os.path.join(dirname, '../homework/deep'))
			self.m.load_state_dict(checkpoint)
		except FileNotFoundError:
			print ("[E] Could not found checkpoint, please make sure you train your deep model first")
			return

		with self.SECTION("Testing classification accuracy of deep classifier with 10000 samples"):
			for threshold in np.arange(0.95, 0.98, 0.01):
				for it in range(1):
					with self.GROUP():
						inputs = np.random.uniform(size=[10000, 2])
						labels = (np.sum(inputs * inputs, axis=1) > 1.0).astype('float')
						linear_out = self.m(torch.tensor(inputs).float()).detach().numpy()
						linear_out = np.argmax(linear_out, axis=1)
						eq = (labels == linear_out).astype('float')
						accuracy = np.mean(eq)
						self.CASE(accuracy > threshold)
						self.CASE(accuracy < 1.0)