from .main import MainLinear, MainDeep
import os
import matplotlib.pyplot as plt
import numpy as np
import torch

linear_model = MainLinear()
deep_model = MainDeep()

dirname = os.path.dirname(os.path.abspath(__file__))

f, [ax1, ax2] = plt.subplots(1,2, figsize=(10,5))
X, Y = np.meshgrid(np.arange(0,1,0.02),np.arange(0,1,0.02))
inputs = np.array([X.flatten(),Y.flatten()]).T

try:
	checkpoint = torch.load(os.path.join(dirname, 'linear'))
	linear_model.load_state_dict(checkpoint)
	outputs = linear_model(torch.tensor(inputs, dtype=torch.float)).detach().numpy()
	pos = inputs[outputs[:,0] > 0]
	neg = inputs[outputs[:,0] < 0]
	ax1.scatter(pos[:,0], pos[:,1],s=5)
	ax1.scatter(neg[:,0], neg[:,1],s=5)
	ax1.set_title('Linear Model')

except FileNotFoundError:
	print ("Could not find checkpoint, please make sure you train your linear model first")


try:
	checkpoint = torch.load(os.path.join(dirname, 'deep'))
	deep_model.load_state_dict(checkpoint)
	outputs = deep_model(torch.tensor(inputs, dtype=torch.float)).detach().numpy()
	pos = inputs[outputs[:,0] > 0]
	neg = inputs[outputs[:,0] < 0]
	ax2.scatter(pos[:,0], pos[:,1],s=5)
	ax2.scatter(neg[:,0], neg[:,1],s=5)
	ax2.set_title('Deep Model')
except FileNotFoundError:
	print ("Could not find checkpoint, please make sure you train your linear model first")

plt.show()
