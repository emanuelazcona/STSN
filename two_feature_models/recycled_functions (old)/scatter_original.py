def scatter(x,w):
	
	sampN, featN = x.shape.as_list()	# get input dimensions
	
	# slicing in Tensorflow calls tf.slice automatically because of operator-overloading
	evenMat = x[:,::2]	# take the even-indexed values
	oddMat = x[:,1::2]	# ... .... odd ... ...
	
	'''
		In this function, w takes the shape of a (N x 1) row vector, i.e. wN = 3 (number of trainable weights)
					
				W = [1  1  1]

		Performing this "tile function repeats W for every data "sample" we have so we don't
		have to loop and keep reusing W. Instead we have a matrix of repeating rows, wTile,
		and do the element-wise multiplication directly all at once. Remember that each row 
		refers to each new data sample. If the number of training samples, sampN, is 5, we'll have:

					[1  1  1]
					[1  1  1]
			wTile =	[1  1  1]
					[1  1  1]
					[1  1  1]

		Doing this saves us the headache of using nested for loops and keeps the runtime
		complexity exactly the same unless Tensorflow's backend code has element-wise
		matrix multiplication optimized to run faster than O(n*m)
	'''

	wTile = tf.tile(w, [sampN,1])	# create matrix of repeating rows
	oneMinus = 1 - wTile 			# subtracting the repeating rows matrix by one is the same as tf.ones(.) - wTile
	
	''' !!!!!!!!!!!!!!!!!!!!!!!!!!!!
		CAUTION Do not get confused: 
			- 	tf.multiply(,) is actually element-wise multiplication (that's why I used it)
			- 	tf.matmul(,) is matrix multiplication
			- 	when you do A*B, I'm not sure what Tensorflow's '*' operator is using. In other words,
				we don't have access to see what operator overloading '*' does when A*B is being computed.
	'''

	# top outputs for each "cross" (sorry I couldn't think of better wording)
	top =  tf.multiply(oddMat,wTile) + tf.multiply(evenMat,oneMinus)
	
	# bottom ... ... ... ...
	bottom = tf.multiply(evenMat,wTile) + tf.multiply(oddMat,oneMinus)
	
	# iterate through each feature of the top and bottom matrices
	for i in range(featN//2):

		# if the first iteration is reached, instantiate the output matrix with the first column of top
		if i is 0:
			out = top[:,i]
			out = tf.reshape( out, (sampN,1) )	# must reshape because slicing interprets as 1-D column

		# otherwise just append the next element of top to the already instantiated output
		else:
			nex = top[:,i]
			nex = tf.reshape( nex,(sampN,1) )	# must reshape because slicing interprets as 1-D column

			# every time we obtain the "next" column, append it (I tried using tf.stack (synonymous to np.stack) but it was too confusing)
			out = tf.concat([out, nex], axis = 1)

		# at the end of every iteration, append the i-th element of bottom
		nex = bottom[:,i]
		nex = tf.reshape( nex,(sampN,1) )	# must reshape because slicing interprets as 1-D column
		out = tf.concat([out, nex], axis = 1)

	return out