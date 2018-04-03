import tensorflow as tf

def costMask(x,start,end):		# omit the masked region from the cost (have no info. of output in the material)
	
	a = start*2 - 1		# forward part of wave associated to weight at index: start
	b = end*2			# backward part of wave associated to weight at index: end

	sampN, featN = x.shape.as_list()

	# create matrix of ones and zeros where we purposely multiply by zero (element-wise)
	# to remove certain columns of x

	outTop = tf.ones(shape = [sampN,a-1], dtype = tf.float64)			# columns that are 1
	outMid = tf.zeros(shape = [sampN,b-a+1], dtype = tf.float64)		# columns that are 0
	outBottom = tf.ones(shape = [sampN,featN-b], dtype = tf.float64)	# columns that are 1

	# concatenate these parts to create a matrix that zeros out parts of x
	out = tf.concat([outTop, outMid, outBottom], axis = 1)

	# return x with parts of itself being zeroed out
	return tf.multiply(out,x)