import pickle

from matplotlib import pyplot as plt

import numpy as np

def save_data(folder, name, i, status):
	# save data for "i-th" epoch to pickle file
	name = "/epoch" + str(i) + "_lossAndWeights.p"
	pickle.dump( status, open( folder + name, "wb" ) )

def print_data(i, loss, weights):
	# print information for the user about loss and weights
	print("Epoch: " + str(i) + "\t\tLoss: " + str(loss))
	print(weights)


plt.ion()
global cbar
cbar = None
fig = plt.figure(1)
ax = fig.gca()
plt.title("Static Weights per Time Unit")
ax.set_xlabel('Time Unit')
ax.set_ylabel('Weight Index')

def plot_data(epoch, loss, weights, time):
	global cbar

	W = weights.T
	wN = np.shape(W)[0]
	W = np.tile(W, (1,time))
	im = plt.imshow(W,
					cmap = plt.cm.jet, 
					interpolation='none',
					aspect="auto",
					extent = [1,time, wN, 1])
	
	plt.title("Static Weights per Time Unit\nEpoch: " + str(epoch) + ",\t Loss: " + str(loss))
	if cbar:
		cbar.update_bruteforce(im)
	else:
		cbar = plt.colorbar(im)
	plt.pause(5e-3)