import argparse, pickle
import numpy as np
import os
from itertools import cycle

import torch
from torch import nn, optim

from .main import model
from .models import *

import ray
from .policy_eval import PolicyEvaluator

ray.init()
dirname = os.path.dirname(os.path.abspath(__file__))


def train(epoch):
    '''
    This is the main training function. You need to fill in this function to complete the assignment
    '''


    levels = {
        '01 - Welcome to Antarctica.stl' : 0.2,
        '02 - The Journey Begins.stl': 0.2,
        '03 - Via Nostalgica.stl' : 0.2,
        '04 - Tobgle Road.stl' : 0.2,
        '05 - The Somewhat Smaller Bath.stl' : 0.2,
    }

    print (model)

    print ("Num of parameters: ", sum([p.numel() for p in model.parameters() if p.requires_grad]))

    for t in range(epoch):

        '''
        Your code here : use your favorite method here
        '''

        '''
        Your code here : optionally, you can print diagnostics of your model below
        '''

        rewards = [0] # Replace me with actual rewards

        # Print diagnostics
        print ('====== Iter: %d ======' % t)
        print ("Mean reward: %.5f" % np.mean(rewards))
        print ("Std reward: %.5f" % np.std(rewards))
        print ("Min reward: %.5f" % np.min(rewards))
        print ("Max reward: %.5f" % np.max(rewards))


    # Save model
    torch.save(model.state_dict(), os.path.join(dirname, 'model.th')) # Do NOT modify this line
    



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--epoch', type=int, default=50)

    args = parser.parse_args()

    print ('[I] Start training')
    train(args.epoch)
    print ('[I] Training finished')
