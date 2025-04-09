import tensorflow as tf
import numpy as np

print(tf.version.VERSION)

#creating some tensors
string = tf.Variable("this is a string", tf.string)
number = tf.Variable(324, tf.int16)
floating = tf.Variable(3.567, tf.float64)
rank1_tensor = tf.Variable(["Test"], tf.string)
rank2_tensor = tf.Variable([["test", "ok"], ["test", "yes"]], tf.string)

#basic tensor opetrations
print(tf.rank(rank2_tensor))
print(rank2_tensor.shape)
tensor1 = tf.ones([1, 2, 3])
print(tensor1)
tensor2 = tf.reshape(tensor1, [2, 3, 1])
print(tensor2)
#reshaping tensor to known shape (=6)
tensor3 = tf.reshape(tensor2, [3, -1])
tensor4 = tf.zeros([5, 5, 5])
#flattening prev tensor4
tensor5 = tf.reshape(tensor4, [-1]) #-1 -> 625
print(tensor5)

#evaluating tensors
with tf.Session() as sess:
    tensor3.eval()