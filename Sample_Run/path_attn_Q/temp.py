from __future__ import  print_function
import tensorflow as tf
import numpy as np

sess = tf.Session()

data = tf.constant(np.ones((4,2,3)))
lengths = tf.constant([3,2])

cell = tf.nn.rnn_cell.LSTMCell(5)
outs, states = tf.nn.dynamic_rnn(cell, data, sequence_length=lengths, dtype=tf.float64, time_major = True)

sel = outs[2][0]
sess.run(tf.global_variables_initializer())
print(sess.run([outs, states, sel]))