from .main import Main
import torch

m = Main()
torch.manual_seed(1234)
for k in [4,16,64,256,1024,543]:
	print( 'Example [%5d x 2 numbers]'%k, m(torch.rand([k,2], dtype=torch.float32)).numpy() )
