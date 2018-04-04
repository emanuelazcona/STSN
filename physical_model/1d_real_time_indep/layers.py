import tensorflow as tf
import numpy as np
from scipy.sparse import block_diag

# ---------------------- Transfer Matrix Generation ----------------------#
def create_transfer_matrix(featN):
	
	transfer = np.zeros((featN,featN))

	wN = featN//3

	for j in range(wN):
		if j >= 1:
			transfer[3*j-2, 3*j] = 1

		if j <= wN-2:
			transfer[3*j + 3, 3*j + 1] = 1

		transfer[3*j+2, 3*j+2] = 1

	return tf.convert_to_tensor(transfer, dtype = tf.float64)



# ---------------------- Scatter Matrix Generation ----------------------#
def create_scatter_matrix(w):

	N = w.shape.as_list()[0]	# extract number of weights, N

	A = [ [-1,0,2] , [0,-1,2] , [0,0,1] ]				# initialize A matrix
	A = block_diag([A for _ in range(N)]).toarray()		# block diagonalize N times
	A = tf.constant(A, dtype = tf.float64)


	B = [ [0,2,0] , [2,0,0] , [2,2,-2] ]				# initialize B matrix
	B = block_diag([B for _ in range(N)]).toarray()		# block diagonalize N times


	# replicates the weights N times into a 1D vector (i.e. [1,2,3] --> [1,1,1,2,2,2,3,3,3] if N = 3)
	Y = [tf.diag_part(w[i]*tf.eye((3), dtype = tf.float64)) for i in range(N)]
	Y = tf.stack(Y)
	Y = tf.reshape(Y, [-1])

	alpha = 1/ (2 + Y)		# compute alpha (function of Y)
	alpha = tf.diag(alpha)	# diagonalize vector alpha into diagonal matrix

	Y = tf.diag(Y)			# diagonalize vector of replicated weights

	scatter = tf.matmul( tf.matmul(Y,A) + B , alpha )	# create scatter matrix for 1 time unit

	return scatter



#-------------------------- Transmision ------------------------#
def predict(x, scatter, transfer, N):

	for i in range(1, N+1):
		if i is 1:
			out = x

		out = tf.matmul(transfer , tf.matmul(scatter, out) )

	return out



def mask(x,start,end):		# omit the masked region from the cost (have no info. of output in the material)
	
	a = start*3 - 2		# forward part of wave associated to weight at index: start
	b = end*3			# backward part of wave associated to weight at index: end

	featN,sampN = x.shape.as_list()

	# create matrix of ones and zeros where we purposely zero-out (element-wise) for certain rows of x

	outTop = tf.ones(shape = [a-1,sampN], dtype = tf.float64)			# columns that are 1
	outMid = tf.zeros(shape = [b-a+1,sampN], dtype = tf.float64)		# columns that are 0
	outBottom = tf.ones(shape = [featN-b,sampN], dtype = tf.float64)	# columns that are 1

	# concatenate these parts to create a matrix that zeros out parts of x
	out = tf.concat([outTop, outMid, outBottom], axis = 0)

	# return x with parts of itself being zeroed out
	return tf.multiply(out,x)