import pickle

def save_data(folder, name, i, status):
	# save data for "i-th" epoch to pickle file
	name = "/epoch" + str(i) + "_lossAndWeights.p"
	pickle.dump( status, open( folder + name, "wb" ) )

def print_data(i, loss, weights):
	# print information for the user about loss and weights
	print("Epoch: " + str(i) + "\t\tLoss: " + str(loss))
	print(weights)