from __future__ import print_function
import numpy as np
import tensorflow as tf

# Create input data
X = np.random.randn(5, 2, 4)

# The second example is of length 6
X[1, 3:] = 0
X_lengths = [5, 3]

x = tf.placeholder(tf.float32, shape=[5, None, 4], name='Input')
x_lengths = tf.placeholder(tf.float32, shape=[None], name='lengths')

cell = tf.nn.rnn_cell.GRUCell(num_units=6)

outputs, last_states = tf.nn.dynamic_rnn(
    cell=cell,
    dtype=tf.float32,
    sequence_length=x_lengths,
    inputs=x,
    time_major=True)

with tf.Session() as sess:
    sess.run( tf.global_variables_initializer())
    out, last = sess.run([outputs, last_states], feed_dict={x:X, x_lengths:X_lengths})

print(np.shape(out), out)
print(np.shape(last), last)

