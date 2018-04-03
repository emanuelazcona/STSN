import os
import pickle
import numpy as np
from scipy.io import savemat

scatter = str(input("Number of Scatter: "))						# user specify number of scatter locations
timeUnits = str(input("\nNumber of Time Units: "))				# ... ... ... ... ... ...time units

maskStart = int(input("\nStarting Weight Index of Mask: "))		# ... ... ... ... ... ... starting scatter index of mask (start at 1)
maskEnd = int(input("\nEnding Weight Index of Mask: "))			# ... ... ... ... ... ... ending ... ... ... ... ... ...  (ends at 100 for our example)

lr = float(input("Learning rate to 3-decimal precision: "))		# learning rate specification (3-decimal digit precision)

# if the scatter index is a 1-digit string, add the 0 to the beginning of the string
if len(scatter) is 1:
	scatter = "0" + scatter

# repeat above if statement to the input string for time units
if len(timeUnits) is 1:
	timeUnits = "0" + timeUnits

# folder containing the information for a specific model
filePath = "results/n" + scatter + "_T" + timeUnits + "_Mask_" + str(maskStart) + "_" + str(maskEnd) + "_lr{0:.3f}".format(lr)

losses = []			# list of loss per epoch (length = number of epochs in training)
weightMatrix = []	# ... ... weights at each time unit per epoch ... ... ... ... ...

# starting epoch number
epoch = 1

# file names starts at epoch1_lossAndWeights.p
fileName = "epoch" + str(epoch) + "_lossAndWeights.p"

# concatenate file path and file name to obtain full filename
fname = filePath + "/" + fileName

# loop until there are no more files with the similar name structure (starting at epoch1_lossAndWeights.p)
while( os.path.isfile(fname) ):

	# "un-pickle" the loos and weights of the current epoch for this model
	currLoss_andWeights = pickle.load( open( fname, "rb" ))

	# loss is the first element of the un-pickled list. Append it to the list of losses
	losses.append(currLoss_andWeights[0])

	# weights are the rest of the elements in the un-pickled
	weightMatrix.append([w for w in currLoss_andWeights[1:]])

	epoch += 1

	# load next file name to check if it is a file in the current directory
	fileName = "epoch" + str(epoch) + "_lossAndWeights.p"
	fname = filePath + "/" + fileName

# matlab model file path
matlab_Path = "results/matlab_models"

# check to see if the path already exists, if not, create it
if not os.path.exists(matlab_Path):
	os.makedirs(matlab_Path)

# create matlab file name using path and the model name
matlab_File = matlab_Path + "/" + filePath[8:]

# save matlab file (.MAT format)
savemat(matlab_File, {'losses': losses, 'weightMatrix': np.asarray(weightMatrix)})
