from torch import nn
import torch.nn.functional as F
import torch

def one_hot(x, n=6):
	batch_size, h, w= x.size()
	x = (x.view(-1,h,w,1) == torch.arange(n, dtype=x.dtype, device=x.device)[None]).float() - torch.as_tensor([0.6609, 0.0045, 0.017, 0.0001, 0.0036, 0.314], dtype=torch.float, device=x.device)
	x = x.permute(0,3,1,2)
	return x


class FConvNetModel(nn.Module):

	"""
	Define your fully convolutional network here
	"""

	def __init__(self):

		super().__init__()


		self.conv1 = nn.Conv2d(9,16,kernel_size=5,stride=1,padding=1, bias = False)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16,32,kernel_size=4,stride=2,padding=0, bias = False)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,64,kernel_size=6,stride=1,padding=0, bias = False)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1, bias = False)
		self.bn4 = nn.BatchNorm2d(128)

		self.upconv1 = nn.ConvTranspose2d(128,64,kernel_size=5,stride=2,padding=1)

		self.upconv2 = nn.ConvTranspose2d(64,32,kernel_size=6,stride=1,padding=0,out)
		self.upconv3 = nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=0)
		self.upconv4 = nn.ConvTranspose2d(16,3,kernel_size=5,stride=1,padding=1)

		nn.init.constant_(self.upconv4.weight, 0)
		nn.init.constant_(self.upconv4.bias, 0)

		# self.finalconv = nn.Conv2d(6,3,2,2,1)

		self.relu = nn.LeakyReLU(inplace=True)
	def forward(self, image, labels):

		'''
		Your code here
		'''
		hr_image = F.interpolate(image, scale_factor=4, mode='bilinear',align_corners=True)
		labels = one_hot(labels)
		x = torch.cat((hr_image, labels), 1)
		c1 = F.leaky_relu(self.bn1(self.conv1(x)),0.2)
		c2 = F.leaky_relu(self.bn2(self.conv2(c1)),0.2)
		c3 = F.leaky_relu(self.bn3(self.conv3(c2)),0.2)
		c4 = F.leaky_relu(self.bn4(self.conv4(c3)),0.2)

		up1 = self.upconv1(c4)
		up2 = self.upconv2(up1+c3)
		up3 = self.upconv3(up2+c2)
		y = self.upconv4(up3+c1)
		return y + hr_image
