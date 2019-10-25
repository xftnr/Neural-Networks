import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import load
from .models import *
from torchvision import transforms

dirname = os.path.dirname(os.path.abspath(__file__))

val_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.429, 0.505, 0.517], std=[0.274, 0.283, 0.347])
])

def accuracy(outputs, labels):
    outputs_idx = outputs.max(1)[1].type_as(outputs)
    return outputs_idx.eq(labels.float()).float().mean()

all_transforms = transforms.Compose([
    transforms.ToPILImage(),

    # """
    # Apply torchvision.transforms to your desire.
    # =====================
    # WRITE YOUR CODE HERE
    # =====================
    # """
    # transforms.Resize(64),
    transforms.RandomHorizontalFlip(),
    # transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.429, 0.505, 0.517], std=[0.274, 0.283, 0.347])
])

def augment_data(inputs):

    """
    ** DO NOT MODIFY THIS FUNCTION **
    """
    return torch.stack( [ all_transforms(inp_i) for inp_i in inputs ] )

def transform_val(val_inputs):

    """
    During Evaluation we don't use data augmentation, we only Normalize the images
    """

    return torch.stack( [ val_transform(inp_i) for inp_i in val_inputs ])

def train(iterations, batch_size=64, log_dir=None):
    '''
    This is the main training function, feel free to modify some of the code, but it
    should not be required to complete the assignment.
    '''

    """
    Load the training data
    """

    train_inputs, train_labels = load(os.path.join('tux_train.dat'))
    val_inputs, val_labels = load(os.path.join('tux_valid.dat'))

    model = ConvNetModel()

    log = None
    if log_dir is not None:
        from .utils import SummaryWriter
        log = SummaryWriter(log_dir)

    # We use the ADAM optimizer with default learning rate (lr)
    # If your model does not train well, you may swap out the optimizer or change the lr

    """
    ==========================================
    Enable L2 regularization of weights below
    ===========================================
    """

    # To enable L2 regularization of weights used in the model, vary the weight_decay the parameter in the optimizer below
    optimizer = optim.Adam(model.parameters(), lr = 1e-3, weight_decay=0.0)

    loss = nn.CrossEntropyLoss()

    for iteration in range(iterations):
        model.train()
        # Construct a mini-batch
        batch = np.random.choice(train_inputs.shape[0], batch_size)

        # Perform data augmentation.
        # Only needed during training
        batch_inputs = augment_data(train_inputs[batch])

        # print (batch_inputs.size())
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
            batch = np.random.choice(val_inputs.shape[0], 256)
            batch_inputs = transform_val(val_inputs[batch])
            # print (batch_inputs.size())
            batch_labels = torch.as_tensor(val_labels[batch], dtype=torch.long)
            model_outputs = model(batch_inputs)
            v_acc_val = accuracy(model_outputs, batch_labels)

            print('[%5d]'%iteration, 'loss = %f'%t_loss_val, 't_acc = %f'%t_acc_val, 'v_acc = %f'%v_acc_val)

            if log is not None:
                log.add_scalar('train/loss', t_loss_val, iteration)
                log.add_scalar('train/acc', t_acc_val, iteration)
                log.add_scalar('val/acc', v_acc_val, iteration)

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
