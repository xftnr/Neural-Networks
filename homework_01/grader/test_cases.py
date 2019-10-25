from . import base
import torch

class Grader(base.Grader):
	TOTAL_SCORE = 100
	BLACK_LIST = ['acos', 'asin', 'atan', 'atan2']
	ALLOWED_CONST = [0, 1, 1.0, 2, 4]
	INPUT_EXAMPLE = (torch.rand([10,2]),)

	def io_check(self):
		# Make sure input accepts a list of random numbers
		o = self.m( *self.INPUT_EXAMPLE )
		self.CHECK_SHAPE(o, [], name='output')
	
	def grade(self):
		import numpy as np
		torch.manual_seed(0)
		with self.SECTION("Testing stochasticity"):
			for it in range(5):
				with self.GROUP():
					for g in range(5):
						r = np.std([float(self.m(torch.rand([1,2]))) for it in range(100)])
						self.CASE( r > 5e-3 )

		with self.SECTION("Approximating pi with 100000 samples"):
			for it in range(5):
				with self.GROUP():
					for g in range(5):
						r = float(self.m(torch.rand([100000,2])))
						self.CASE( np.abs(r - np.pi) < 0.05 )

		with self.SECTION("Approximating pi with 10000 samples"):
			for it in range(5):
				with self.GROUP():
					for g in range(5):
						r = float(self.m(torch.rand([10000,2])))
						self.CASE( np.abs(r - np.pi) < 0.05 )

		with self.SECTION("Approximating pi with 1000 samples"):
			for it in range(5):
				with self.GROUP():
					for g in range(5):
						r = float(self.m(torch.rand([1000,2])))
						self.CASE( np.abs(r - np.pi) < 0.15 )
		
		with self.SECTION("Approximating pi with 100 samples (extra)"):
			for it in range(1):
				with self.GROUP():
					for g in range(100):
						r = float(self.m(torch.rand([100,2])))
						self.CASE( np.abs(r - np.pi) < 0.3, score=0.05 )

		with self.SECTION("Approximating pi with 10 samples (extra)"):
			for it in range(1):
				with self.GROUP():
					for g in range(100):
						r = float(self.m(torch.rand([10,2])))
						self.CASE( np.abs(r - np.pi) < 0.3, score=0.05 )
