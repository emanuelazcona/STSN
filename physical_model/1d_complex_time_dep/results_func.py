import pickle

from matplotlib import pyplot as plt

import numpy as np

def save_data(folder, name, i, status):
	# save data for "i-th" epoch to pickle file
	name = "/epoch" + str(i) + "_lossAndWeights.p"
	pickle.dump( status, open( folder + name, "wb" ) )

def print_data(i, loss, weights):
	t = len(weights)-1
	# print information for the user about loss and weights
	print("Epoch: " + str(i) + "\t\tLoss: " + str(loss))

	# peek = (t//2) - (t//2)//2
	peek = 0
	print("Peek @ Weights for T-U: " + str(peek+1))
	print(weights[peek])

	peek = (t//2) + (t//2)//2

	print("Peek @ Weights for T-U: " + str(peek+1))
	print(weights[peek])


plt.ion()
global cbar, ax
cbar = None
fig = plt.figure(1)
ax = fig.gca()
plt.title("Dynamic Weights per Time Unit")
ax.set_xlabel('Time Unit')
ax.set_ylabel('Weight Index')

def plot_data(epoch, loss, weights):
	global cbar,ax

	W = [w.T for w in weights]

	W = np.concatenate(W, axis = 1)

	wN = np.shape(W)[0]
	time = len(weights)

	im = plt.imshow(W,
					cmap = plt.cm.jet, 
					interpolation='none',
					aspect="auto")
	plt.xticks(range(time), rotation = '45', ha = 'center')
	# plt.yticks(range(wN), ha = 'center')

	plt.title("Dynamic Weights per Time Unit\nEpoch: " + str(epoch) + ",\t Loss: " + str(loss))
	if cbar:
		cbar.update_bruteforce(im)
	else:
		cbar = plt.colorbar(im)
	plt.pause(1e-6)