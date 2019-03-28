import tensorflow as tf


print(tf.__version__)
print(tf.__path__)

#import os
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

hello = tf.constant("Hello,tf!")
sess = tf.Session()
print(sess.run(hello))

