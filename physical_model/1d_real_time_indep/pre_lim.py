# for reading in data and Tensorflow can interpret numpy arrays as inputs/outputs
import numpy as np

# clears numpy warning about floats
# import sys
# for i in range(3):    # Add this for loop.
#     sys.stdout.write('\033[F') # Back to previous line.
#     sys.stdout.write('\033[K') # Clear line.

# terminal/system commands
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='3'				# clears Tensorflow CPU for my mac Unix terminal
os.system('cls' if os.name == 'nt' else 'clear')	# clears the terminal window screen (clc equiv. to MATLAB)

# storing weights and associated cost during training
import pickle

from weight_gen import *		# functions for generating weight vector & scatter/prop. matrices

from layers import *

np.random.seed(7)	# seed random number gen. for NumPy (7 is my lucky number)
tf.set_random_seed(7)	# seed Tensorflow random number gen. (7 ...)