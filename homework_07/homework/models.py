from torch import nn

class FConvNetModel(nn.Module):

	"""
	Define your fully convolutional network here
	"""

	def __init__(self):

		super().__init__()

		'''
		Your code here
		'''
		self.conv1 = nn.Conv2d(3,16,kernel_size=5,stride=1,padding=1, bias = False)
		self.bn1 = nn.BatchNorm2d(16)
		self.conv2 = nn.Conv2d(16,32,kernel_size=4,stride=2,padding=0, bias = False)
		self.bn2 = nn.BatchNorm2d(32)
		self.conv3 = nn.Conv2d(32,64,kernel_size=6,stride=1,padding=0, bias = False)
		self.bn3 = nn.BatchNorm2d(64)
		self.conv4 = nn.Conv2d(64,128,kernel_size=4,stride=2,padding=1, bias = False)
		self.bn4 = nn.BatchNorm2d(128)

		self.upconv1 = nn.ConvTranspose2d(128,64,kernel_size=5,stride=2,padding=1)

		self.upconv2 = nn.ConvTranspose2d(64,32,kernel_size=6,stride=1,padding=0)
		self.upconv3 = nn.ConvTranspose2d(32,16,kernel_size=4,stride=2,padding=0)
		self.upconv4 = nn.ConvTranspose2d(16,6,kernel_size=5,stride=1,padding=1)

		self.finalconv = nn.Conv2d(6,6,2,1,1)
		self.relu = nn.ReLU(True)

	def forward(self, x):

		'''
		Your code here
		'''
		# print(x.size())
		c1 = self.conv1(x)
		c1 = self.relu(c1)
		c2 = self.conv2(c1)
		c2 = self.relu(c2)
		c3 = self.conv3(c2)
		c3 = self.relu(c3)
		c4 = self.conv4(c3)
		y = self.relu(c4)

		up1 = self.upconv1(y)
		up2 = self.upconv2(up1+c3)
		up3 = self.upconv3(up2+c2)
		y = self.upconv4(up3+c1)


		return y
