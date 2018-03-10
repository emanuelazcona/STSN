import numpy as np
import pandas as pd
import os
import pickle

from weightGen import *
from layers import transmitTimeDep
from costMask import costMask

os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

np.random.seed(7)		# seeding the random number generator to reproduce identical results
tf.set_random_seed(7)	# seed Tensorflow random numebr generator as well



#------------------------ Read in Data ------------------------#
userScatter = str( input("Scatter-points: ") )
userTime = str( input("Time units: ") )

layers = int(userTime)	# number of scatter/prop. layers to navigate through
expectedScatter = int(userScatter)

if len( userScatter ) is 1:
	userScatter = "0" + userScatter
if len( userTime ) is 1:
	userTime = "0" + userTime

fileName = 'data/scatter' + userScatter + '_T' + userTime + '_tc20_all_'
X = np.transpose( np.genfromtxt(fileName + 'in.csv', delimiter = ',') )
Y = np.transpose( np.genfromtxt(fileName + 'out.csv', delimiter = ',') )

# X = np.ones((15,30)) # FOR TESTING
# Y = np.random.random((15,30)) # FOR TESTING



#------------------------ Extract Number of Layers ------------------------#
# Print Information
print("This model contains:")
print("\t- " + userTime + " time units")
print("\t- " + userScatter + " expected non-one weights\n")

sampN, featN = X.shape	# sampN: number of training samples, featN: features per sample 



#----------- Random Weight Generation For Material -----------#
wN = featN//2	# number of transmission weights
start = int(input("Starting Weight Index of Mask: "))	# starting index (1 <--> wN)
end = int(input("Ending Weight Index of Mask: "))		# ending index (start <--> wN)

print("\t- " + str(wN*layers) + " total weights")
print("\t- " + str((end-start+1)*layers) + " trainable weights out of total weights (masked region)\n")

# extract arrays for trainable & frozen weights
W_left, W_train, W_right = weightCreation(start, end, wN, layers)
W_tens = weightConcat(W_left, W_train, W_right)



#--------------------------- Placeholder Instantiation --------------------------#
X_tens = tf.placeholder(dtype = tf.float64, shape = [sampN,featN])
Y_tens = tf.placeholder(dtype = tf.float64, shape = [sampN,featN])



#--------------------------- Cost Function Definition --------------------------#
# compute least squares cost for each sample and then average out their costs
print("Building Cost Function (Least Squares) ... ... ...")

Yhat_tens = transmitTimeDep(X_tens, W_tens, layers)	# prediction function

Yhat_masked = costMask(Yhat_tens - Y_tens, start, end)	# masking region "we don't know" for the cost

# perform least squares by squaring l2-norm (normalizing the cost by the number of known points)
least_squares = tf.norm(Yhat_masked, ord=2)**2 / (featN - ((end*2) - (start*2-1) + 1))	#
print("Done!\n")



#--------------------------- Define Optimizer --------------------------#
print("Building Optimizer ... ... ...")
lr = float(input("Specify learning-rate of the model (3-decimal precision): "))
train_op = tf.train.AdamOptimizer(learning_rate=lr).minimize(least_squares, var_list = [W_train])
print("Done!\n")



#--------------------------- Training --------------------------#
epochs = 2000
loss_tolerance = 1e-8

with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
	sess.run( tf.global_variables_initializer() )
	
	print("Tensor X:")		# show info. for X
	print(X)
	print("")

	print("Tensor W: ") 	# show info. for W total
	print(W_tens.eval())
	print("")

	# show only the trainable part of W
	print("Trainable part of W (weights " + str(start) + " through " + str(end) + "):")
	print(W_train.eval())
	print("")

	print("Tensor Y: ")		# show info. for Y
	print(Y)
	print("")

	print("--------- Starting Training ---------\n")
	for i in range(1, epochs+1):

		# run X and Y dynamically into the network per iteration
		_, loss_value = sess.run([train_op, least_squares], feed_dict = {X_tens: X, Y_tens: Y})
		
		W_tens = tf.clip_by_value(W_tens, 0.0, 1.0)	# after updating the weights clip them to stay between 0 and 1
		currStatus = [loss_value]	# status of the network for the current epoch

		# add the weights to the status of the network for current epoch
		for j in range(layers):
			currStatus.append(W_tens.eval())

		# saves objects for every iteration
		fileFolder = "results/n" + userScatter + "_T" + userTime + "_Mask_" + str(start) + "_" + str(end) + "_lr{0:.3f}".format(lr)

		if not os.path.exists(fileFolder):
			os.makedirs(fileFolder)
		fileName = "/epoch" + str(i) + "_lossAndWeights.p"
		pickle.dump( currStatus, open( fileFolder + fileName, "wb" ) )

		# print information for the user about loss and weights
		print("Epoch: " + str(i) + "\t\tLoss: " + str(loss_value))
		print("Peek Weights @ Time Unit 15: ")
		print(W_tens[15,:].eval())
		print("\nPeek Weights @ Time Unit 25: ")
		print(W_tens[25,:].eval())

		if loss_value <= loss_tolerance:
			break