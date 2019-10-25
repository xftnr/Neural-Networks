import argparse, pickle
import numpy as np
import os

import torch
from torch import nn, optim

# Use the util.load() function to load your dataset
from .utils import load
from .models import *
from .train import CM, J, one_hot

import matplotlib.pyplot as plt

dirname = os.path.dirname(os.path.abspath(__file__))

COLORS = np.array([
	(0,0,0),
	(255,0,0),
	(0,255,0),
	(255,255,0),
	(0,0,255),
	(255,255,255)], dtype=np.uint8)

def test(batch_size=5):
	valid_dataloader = load('valid', num_workers=4, batch_size=batch_size)
	valid_dataloader_iterator = iter(valid_dataloader)

	fig, axes = plt.subplots(batch_size,4,figsize=(4*2,batch_size*2))
	valid_batch = next(valid_dataloader_iterator)
	batch_inputs = valid_batch['inp_image']
	batch_labels = valid_batch['lbl_image'].long()

	model = FConvNetModel()
	model.load_state_dict(torch.load(os.path.join(dirname, 'fconvnet.th'), map_location=lambda storage, loc: storage))
	model.eval()
	
	pred = model(batch_inputs).argmax(dim=1)

	for i, (image, label, p) in enumerate(zip(batch_inputs, batch_labels, pred)):
		image = np.transpose(image.detach().numpy(), [1,2,0])
		axes[i,0].imshow(np.clip(image*[0.246,0.286,0.362]+[0.354,0.488,0.564], 0, 1))
		axes[i,1].imshow(COLORS[label])
		axes[i,2].imshow(COLORS[p])
		axes[i,3].imshow(label == p)
		
	
	# Compute the confusion matrix
	plt.figure()
	CMs = []
	for it in range(50):
		valid_batch = next(valid_dataloader_iterator)
		batch_inputs = valid_batch['inp_image']
		batch_labels = valid_batch['lbl_image'].long()

		model_outputs = model(batch_inputs)
		model_pred = torch.argmax(model_outputs, dim=1)
		CMs.append( CM(model_pred, batch_labels).numpy() )
	
	cm = np.mean(CMs, axis=0)
	cm = cm / np.max(cm)
	plt.imshow(cm, interpolation='nearest')
	plt.title('IoU = %f'%float(J(torch.as_tensor(cm))))
	for j in range(cm.shape[0]):
		for i in range(cm.shape[1]):
			plt.text(i, j, '%0.3f'%cm[j, i], horizontalalignment="center", color="black")
	
	plt.show()

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--batch_size', type=int, default=5)
	args = parser.parse_args()

	print ('[I] Testing')
	test(args.batch_size)
