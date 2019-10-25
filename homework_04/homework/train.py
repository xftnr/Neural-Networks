import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import load
from .models import *

dirname = os.path.dirname(os.path.abspath(__file__))

 
def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(outputs)
    return outputs_idx.eq(labels.float()).float().mean()

def train(iterations, batch_size=64, log_dir=None):
    '''
    This is the main training function, feel free to modify some of the code, but it
    should not be required to complete the assignment.
    '''

    """
    Load the training data
    """

    train_inputs, train_labels = load(os.path.join('tux_train.dat'))
    
    model = ConvNetModel()

    log = None
    if log_dir is not None:
        from .utils import SummaryWriter
        log = SummaryWriter(log_dir)
    
    # We use the ADAM optimizer with default learning rate (lr)
    # If your model does not train well, you may swap out the optimizer or change the lr
    optimizer = optim.Adam(model.parameters(), lr = 1e-3)
    loss = nn.CrossEntropyLoss()
    
    for iteration in range(iterations):
        model.train()
        # Construct a mini-batch
        batch = np.random.choice(train_inputs.shape[0], batch_size)
        batch_inputs = torch.as_tensor(train_inputs[batch], dtype=torch.float32)
        batch_inputs = batch_inputs.permute(0, 3, 1, 2)
        batch_labels = torch.as_tensor(train_labels[batch], dtype=torch.long)

        
        # zero the gradients (part of pytorch backprop)
        optimizer.zero_grad()
        
        # Compute the model output and loss (view flattens the input)
        model_outputs = model(batch_inputs)
        t_loss_val = loss(model_outputs, batch_labels)
        t_acc_val = accuracy(model_outputs, batch_labels)


        # Compute the gradient
        t_loss_val.backward()
        
        # Update the weights
        optimizer.step()
        
        if iteration % 10 == 0:
            model.eval()

            print('[%5d]'%iteration, 'loss = %f'%t_loss_val, 'acc = %f'%t_acc_val)
            if log is not None:
                log.add_scalar('train/loss', t_loss_val, iteration)
                log.add_scalar('train/acc', t_acc_val, iteration)

    # Save the trained model
    torch.save(model.state_dict(), os.path.join(dirname, 'convnet.th')) # Do NOT modify this line

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--iterations', type=int, default=10000)
    parser.add_argument('-l', '--log_dir')
    args = parser.parse_args()

    print ('[I] Start training')
    train(args.iterations, log_dir=args.log_dir)
    print ('[I] Training finished')
