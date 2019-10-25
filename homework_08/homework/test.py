import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import load
from .models import *
from .train import downsample

import matplotlib.pyplot as plt

dirname = os.path.dirname(os.path.abspath(__file__))

COLORS = np.array([
	(0,0,0),
	(255,0,0),
	(0,255,0),
	(255,255,0),
	(0,0,255),
	(255,255,255)], dtype=np.uint8)


def get_rgb(batch_images):

	channel_1 = (batch_images[:, 0, :, :] * 0.246 + 0.354)
	channel_2 = (batch_images[:, 1, :, :] * 0.286 + 0.488)
	channel_3 = (batch_images[:, 2, :, :] * 0.362 + 0.564)

	return torch.clamp(torch.stack((channel_1, channel_2, channel_3), 1), 0, 1)

def test(batch_size=5):
	valid_dataloader = load('valid', num_workers=4, batch_size=batch_size)
	valid_dataloader_iterator = iter(valid_dataloader)

	fig, axes = plt.subplots(batch_size,3,figsize=(batch_size*3,batch_size))
	valid_batch = next(valid_dataloader_iterator)
	batch_targets = valid_batch['inp_image']
	batch_labels = valid_batch['lbl_image'].long()
	batch_inputs = downsample(batch_targets)
	loss = nn.L1Loss()

	model = FConvNetModel()
	model.load_state_dict(torch.load(os.path.join(dirname, 'fconvnet.th'), map_location=lambda storage, loc: storage))
	model = model
	model.eval()
	
	pred = get_rgb(model(batch_inputs,  batch_labels).detach())
	batch_targets = get_rgb(batch_targets.detach())
	batch_inputs = get_rgb(batch_inputs.detach())
	
	l1_loss = loss(pred, batch_targets)
	print('L1 Loss = %f'%float(l1_loss.item()*255))

	for i, (image, out_image, in_image) in enumerate(zip(batch_targets, batch_inputs, pred)):
		
		image = np.transpose(image.cpu().numpy(), [1,2,0])
		in_image = np.transpose(in_image.cpu().numpy(), [1,2,0])
		out_image = np.transpose(out_image.cpu().numpy(), [1,2,0])
		axes[i,0].set_title('Ground Truth')
		axes[i,0].imshow(image)
		axes[i,1].imshow(in_image)
		axes[i,1].set_title('Model Output')
		axes[i,2].imshow(out_image)
		axes[i,2].set_title('Input')
		
	plt.show()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--batch_size', type=int, default=5)
	args = parser.parse_args()

	print ('[I] Testing')
	test(args.batch_size)
