import tensorflow as tf

# ---------------------- Propagation Layer ----------------------#
def propagate(x):
	sampN, featN = x.shape.as_list()	# get shape of the data, O(1) constant run-time

	# each iteration of this for-loop performs an O(n) slice along a column, therefore
	# total run-time is O(200n) = O(n)
	for i in range(featN):
		
		# if the first feature, transmit nothing
		if i is 0:
			out = tf.zeros(shape = [sampN,1], dtype = tf.float64)
			# to avoid concatenation, we have to continue to the next iteration
			continue
		
		# if the last feature, transmit nothing
		elif i is featN-1:
			nex = tf.zeros(shape = [sampN,1], dtype = tf.float64)
			
		# otherwise, if at odd index, transmit input from southwest (according to figure)
		elif i % 2:
			nex = x[:,i+1]
			nex = tf.reshape( nex,(sampN,1) )	# must reshape because slicing interprets as 1-D column
			
		# else if at even index, transmit input from northwest (according to figure)
		else:
			nex = x[:,i-1]
		
		# must reshape because slicing interprets as 1-D column
		nex = tf.reshape( nex,(sampN,1) )

		# every time we obtain the "next" column, append it (tried using tf.stack, too confusing)
		out = tf.concat([out, nex], axis = 1)
	
	return out



#------------------------ Scatter Layer ------------------------#
def scatter(x,w):
	sampN, featN = x.shape.as_list() # get shape of the data, O(1) constant run-time

	for j in range(featN//2):
		# extract current weight
		curr_w = w[0,j]
		
		# create matrix for scatter computation
		W_mat = tf.convert_to_tensor([[1-curr_w,curr_w],[curr_w,1-curr_w]], dtype = tf.float64)

		# much cleaner
		if j is 0:
			out = tf.matmul( x[:, j*2:(j*2)+2], W_mat )
		else:
			nextNode = tf.matmul( x[:, j*2:(j*2)+2], W_mat )
			out = tf.concat([out, nextNode], axis = 1)

	return out



#-------------------------- Transmision ------------------------#
def transmit(x, w, N):

	# for N time units, run X through a network of N time units
	for i in range(1,N+1):

		# input to first time unit is x, otherwise it's the output of the previous iteration
		if i is 1:
			out = x

		out = scatter(out,w)	# scatter portion of time unit
		out = propagate(out)	# propagation portion of next time unit

	return out

# complex version of transmit that 
# applies transmit() to the real and imag.
# parts of x seperately
def transmitComplex(x,w,N):
	sampN, featN = x.shape.as_list()

	xReal = x[:,::2]	# extract real part of x
	xImag = x[:,1::2]	# ... ... imag. ... ....

	yReal = transmit(xReal, w, N)	# transmit real component
	yImag = transmit(xImag, w, N)	# share weights imag. ...

	# create concatenated output with real & imaginary components
	for i in range(featN//2):

		# if first sample, initiate starting with real forward component
		if i is 0:
			out = yReal[:,i]
			out = [ tf.reshape( out,(sampN,1) ) ] # create list of tensors to concat.

		# otherwise continue with real/imag alternation
		else:
			nex = yReal[:,i]
			nex = tf.reshape( nex,(sampN,1) )
			out.append(nex)	# append to list of tensors for concat.

		nex = yImag[:,i]
		nex = tf.reshape( nex,(sampN,1) )
		out.append( nex )	# append to list of tensors for concat.

	return tf.concat(out, axis = 1) # return matrix (concatenated vectors)