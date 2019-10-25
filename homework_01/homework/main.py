import torch

class Main(torch.nn.Module):
	def forward(self, x):
		# The input x is a series of random numbers of size k x 2
		# You should use these random numbers to compute and return pi using pytorch
		# dist1 = torch.sqrt(x[:,0]**2 + x[:,1]**2)
		# dist2 = torch.sqrt((x[:,0]-1)**2+x[:,1]**2)
		# dist3 = torch.sqrt((x[:,0]-1)**2+(x[:,1]-1)**2)
		# dist4 = torch.sqrt(x[:,0]**2+(x[:,1]-1)**2)
		# inarea1 = torch.trunc(dist1)
		# inarea2 = torch.trunc(dist2)
		# inarea3 = torch.trunc(dist3)
		# inarea4 = torch.trunc(dist4)
		# total = torch.mean(1-inarea1)+torch.mean(1-inarea2)+torch.mean(1-inarea3)+torch.mean(1-inarea4)
		# x = I
		list_pi = []
		for i in range(20):
		  # Estimate pi using both x and 1-x
		  pi = (torch.mean(torch.sqrt(1-x*x)) + torch.mean(torch.sqrt(2-x*x)))/2
		  list_pi.append(pi)
		  # Remove the most significant bit (and use the lower bits)
		  x = x * 2
		  x -= torch.floor(x)
		# Pi the the average of all bit-shifted versions in list_pi
		pi = torch.mean(list_pi)
		return total
