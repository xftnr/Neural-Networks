import torch
from torch import nn

# Unlike previous classes, don't use these classes directly in train.py
# Use the functions given in main.py
# However, model definition still needs to be defined in these classes

class ConvNetModel(nn.Module):
    '''
        Your code for the model that computes classification from the inputs to the scalar value of the label.
        Classification Problem (1) in the assignment
    '''

    def __init__(self):
        super().__init__()
        '''
        Your code here
        '''

        self.dim = 64*64*3
        # 3380
        self.fc1 = nn.Linear(180, 100)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(100,6)

        self.con1 = nn.Conv2d(3, 33, 4, stride=2)
        self.con2 = nn.Conv2d(33, 33, 3, stride=2)
        self.con3 = nn.Conv2d(33, 20, 5)

        self.maxpool = nn.MaxPool2d(2)



    def forward(self, x):
        '''
        Input: a series of N input images x. size (N, 64*64*3)
        Output: a prediction of each input image. size (N,6)
        Your code here
        '''

        temp = self.con1(x)
        temp = self.relu(temp)
        temp = self.con2(temp)
        temp = self.maxpool(temp)
        temp = self.con3(temp)

        # print(self.xx(temp))
        temp = temp.contiguous().view(-1, 180)

        x = self.fc1(temp)
        x = self.relu(x)
        x = self.fc2(x)
        return x

    # def xx(self, x):
    #     size = x.size()[1:]
    #     temp = 1
    #     for i in size:
    #         temp *=i
    #     return temp
