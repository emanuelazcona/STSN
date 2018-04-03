# for reading in data and Tensorflow can interpret numpy arrays as inputs/outputs
import numpy as np

# terminal/system commands
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

# storing weights and associated cost during training
import pickle

from weight_gen import *		# functions for generating weight vector & scatter/prop. matrices
from layers import predict		# estimated predict function (error minimized using MSE)

np.random.seed(7)	# seed random number gen. for NumPy (7 is my lucky number)
tf.set_random_seed(7)	# seed Tensorflow random number gen. (7 ...)



#------------------------ Read in Data ------------------------#
user_scatter = str( input("Number of scatter points: ") )
user_time = str( input() )