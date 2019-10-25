import torch
from torch import nn

# Unlike previous classes, don't use these classes directly in train.py
# Use the functions given in main.py
# However, model definition still needs to be defined in these classes

class Block(nn.Module):
    '''
    Your code for resnet blocks
    '''
    def __init__(self, in_channel, bottle_channel, out_channel, stride):
        super(Block, self).__init__()
        if bottle_channel != 0:
            self.conv1 = nn.Conv2d(in_channel, bottle_channel, kernel_size=1, stride=1, padding=0)
            self.bn1 = nn.BatchNorm2d(bottle_channel)
            self.conv2 = nn.Conv2d(bottle_channel,bottle_channel, kernel_size=3, stride=stride,padding=1)
            self.bn3 = nn.BatchNorm2d(bottle_channel)
            self.conv3 = nn.Conv2d(bottle_channel,out_channel, kernel_size=1, stride=1, padding=0)
            self.bn2 = nn.BatchNorm2d(out_channel)
        else:
            self.conv1 = nn.Conv2d(in_channel, 48, kernel_size=1, stride=1, padding=1)
            self.bn1 = nn.BatchNorm2d(48)
            self.conv2 = nn.Conv2d(48,48, kernel_size=3, stride=stride,padding=0)
            self.bn3 = nn.BatchNorm2d(48)
            self.conv3 = nn.Conv2d(48,out_channel, kernel_size=1, stride=1, padding=0)
            self.bn2 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU(True)
        self.residual = nn.Sequential()
        if in_channel != out_channel:
            self.residual.add_module('conv',nn.Conv2d(in_channel, out_channel,kernel_size=1, stride=stride, padding=0))
            self.residual.add_module('bn', nn.BatchNorm2d(out_channel))

        self.out_channel = out_channel

    def forward(self, x):
        '''
        Your code here
        '''
        y = self.conv1(x)
        y = self.bn1(y)
        # y = self.relu(y)
        y = self.conv2(y)
        # y = self.bn1(y)
        # y = self.relu(y)
        y = self.conv3(y)
        y = self.bn2(y)
        y += self.residual(x)
        y = self.relu(y)
        return y




class ConvNetModel(nn.Module):
    '''
    Your code for the model that computes classification from the inputs to the scalar value of the label.
    Classification Problem (1) in the assignment
    '''

    def __init__(self):
        super(ConvNetModel, self).__init__()
        '''
        Your code here
        '''
        self.conv1 = nn.Conv2d(3, 32, 5, 2, 1)
        self.relu = nn.ReLU(True)
        self.conv2 = nn.Conv2d(32, 6, 5, 2, 1)
        self.pool = nn.MaxPool2d(2)
        self.block1 = Block(32,16,64,2)
        self.block2 = Block(64,0,32,1)
        self.block3 = Block(32,8,16,2)
        self.block4 = Block(16,0,8,1)
        self.block5 = Block(8,4,32,1)
    def forward(self, x):
        '''
        Input: a series of N input images x. size (N, 64*64*3)
        Output: a prediction of each input image. size (N,6)
        Your code here
        '''
        x = self.conv1(x)
        x = self.relu(x)
        x = self.block1.forward(x)
        x = self.block2.forward(x)
        x = self.block3.forward(x)
        x = self.block4.forward(x)
        x = self.block5.forward(x)
        x = self.conv2(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return x
