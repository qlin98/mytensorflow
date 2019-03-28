# w b
import tensorflow as tf
import numpy as np

#Prepare
tra_X = np.linspace(-10,10,1000)
tra_Y = 2 * tra_X + np.random.randn(*tra_X.shape) * 1.33 + 10

#Define the model
X = tf.placeholder("float")
Y = tf.placeholder("float")
w = tf.Variable(0.0, name="weight")
b = tf.Variable(0.0, name="bias")
loss = tf.square(Y - (X* w) - b)
train_op = tf.train.GradientDescentOptimizer(0.01).minimize(loss)

#Create session 
with tf.Session() as sess:
	sess.run(tf.initialize_all_variables())
	epoch = 1
	for i in range(50):
		for (x, y) in zip(tra_X, tra_Y):
			_, w_value, b_value = sess.run([train_op, w, b], feed_dict={X: x, Y: y})
		print("Epoch: {}, w:{}, b:{}".format(epoch, w_value, b_value))
		epoch += 1

