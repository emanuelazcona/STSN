from pre_lim import *



#------------------------ Read in Data ------------------------#
user_scatter = str( input("Number of scatter points: ") )
user_time = str( input("Number of time units: ") )

layers = int(user_time)	# number of time units = number of layers in our network

# ensure that the naming convention is always "0#" for single digit inputs
if len(user_scatter) is 1:
	user_scatter = "0" + user_scatter
if len(user_time) is 1:
	user_time = "0" + user_time

fileName = "data/scatter" + user_scatter + "_T" + user_time + "_all_"

# X = np.genfromtxt(fileName + "in.csv", delimiter = ",")	# input data
# Y = np.genfromtxt(fileName + "out.csv", delimiter = ",")	# output data

X = np.random.random((900,246)) # FOR TESTING
Y = np.random.random((900,246)) # FOR TESTING



#------------------------ Extract Number of Samples/Features ------------------------#

# sampN: number of training samples
# featN: 3 times the number of nodes (each node has: forward, back, stub)
featN,sampN = X.shape



#----------- Weight Generation (Always >= 0) -----------#
wN = featN//3	# number of weights

mask_start = int( input("Starting weight index of mask: ") )
mask_end = int( input("Ending weight index of mask: ") )

# extract arrays for trainable & frozen weights
W_left, W_train, W_right = create_weights(mask_start, mask_end, wN)
W_tens = concat_weights(W_left, W_train, W_right)

print("\t- " + str(wN) + " total weights")
print("\t- " + str(mask_end-mask_start+1) + " trainable weights (masked region)\n")

transfer = create_transfer_matrix(featN)
scatter = create_scatter_matrix(W_tens)



#--------------------------- Placeholder Instantiation --------------------------#
X_tens = tf.placeholder(dtype = tf.float64, shape = [featN, sampN])	# placeholder for input
Y_tens = tf.placeholder(dtype = tf.float64, shape = [featN, sampN])	# placeholder for ground truth output


Yhat_tens = predict(X_tens, scatter, transfer, layers)	# placeholder function for prediction



#--------------------------- Cost Function Definition --------------------------#
print("Building Cost Function (Least Squares) ... ... ...")
error = mask(Yhat_tens - Y_tens, mask_start, mask_end)			# placeholder for error
least_squares = tf.norm(error, ord=2)**2
print("Done!\n")



#--------------------------- Define Optimizer --------------------------#
print("Building Optimizer ... ... ...")
lr = 0.05

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
	print("Trainable part of W (weights " + str(mask_start) + " through " + str(mask_end) + "):")
	print(W_train.eval())
	print("")

	print("Tensor Y: ")		# show info. for Y
	print(Y)
	print("")

	print("--------- Starting Training ---------\n")
	for i in range(1, epochs+1):

		# run X and Y dynamically into the network per iteration
		_, loss_value = sess.run([train_op, least_squares], feed_dict = {X_tens: X, Y_tens: Y})
		
		W_tens = tf.clip_by_value(W_tens, 0.0, 10.0)	# after updating the weights clip them to stay between 0 and 1

		# save loss and current weights every 10 epochs
		if i % 10 is 0:
			currStatus = [loss_value, W_tens.eval()]

		# print information for the user about loss and weights
		print("Epoch: " + str(i) + "\t\tLoss: " + str(loss_value))
		print(W_train.eval())

		if loss_value <= loss_tolerance:
			break