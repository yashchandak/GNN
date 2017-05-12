import tensorflow as tf
from tensorflow.contrib.rnn import RNNCell
import numpy as np
from utils import *

def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)


def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer


class myRNN(RNNCell):
    def __init__(self, num_units, x_size):
        self.num_units = num_units
        self.x_size = x_size
        self.W_xh = tf.get_variable('W_xh', [x_size[0], self.num_units])
        self.W_xh_L = tf.get_variable('W_xh_L', [x_size[1], self.num_units])
        self.W_hh = tf.get_variable('W_hh', [self.num_units, self.num_units])
        self.bias = tf.get_variable('bias', [self.num_units])

    @property
    def state_size(self):
        return tuple([self.num_units, self.num_units])

    @property
    def output_size(self):
        return self.num_units

    def candidate_weights_bias(self):
        return self.W_xh, self.bias

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h = state[0]
            weights = tf.concat([self.W_xh, self.W_xh_L], axis=0)
            #hidden = tf.matmul(x, weights) + self.bias
            hidden = tf.matmul(x, weights) + self.bias + tf.matmul(h, self.W_hh)
            new_state = tf.nn.relu(hidden)
            return new_state, (new_state, new_state)


class MyLSTMCell(RNNCell):
    '''Vanilla LSTM implemented with same initializations as BN-LSTM'''

    def __init__(self, num_units, x_size):
        self.num_units = num_units
        self.x_size = x_size
        self.W_xh = tf.get_variable('W_xh', [x_size[0]+x_size[1], 4 * self.num_units], initializer=orthogonal_initializer())
        self.W_hh = tf.get_variable('W_hh', [self.num_units, 4 * self.num_units],
                                    initializer=bn_lstm_identity_initializer(0.95))
        self.bias = tf.get_variable('bias', [4 * self.num_units])  # intializer ???

    def candidate_weights_bias(self):
        weights = tf.slice(self.W_xh[:self.x_size[0]], [0, 0], [-1, self.num_units])
        bias = self.bias[:][:self.num_units]
        return weights, bias

    @property
    def state_size(self):
        return tuple([self.num_units, self.num_units])

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # improve speed by concat.
            data = tf.concat([x, h], 1)  # shape = b*(d+h)
            W_both = tf.concat([self.W_xh, self.W_hh], 0)  # shape = (d+h)*4h
            hidden = tf.matmul(data, W_both) + self.bias  # shape = b*4h

            j, i, f, o = tf.split(hidden, axis=1, num_or_size_splits=4) #j and i changed

            #new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.nn.relu(j)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)


class LSTMgated(RNNCell):
    '''Vanilla LSTM implemented with same initializations as BN-LSTM'''

    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return tuple([self.num_units, self.num_units])

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        x, context = x[0], x[1]
        with tf.variable_scope(scope or type(self).__name__):
            c, h = state

            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            x_size = x.get_shape().as_list()[1]
            c_size = context.get_shape().as_list()[1]
            context_size = context.get_shape().as_list()[1]
            W_g = tf.get_variable('W_g',
                                  [x_size + self.num_units + context_size, 3 * self.num_units],
                                  initializer=orthogonal_initializer())
            b_g = tf.get_variable('W_g', [3 * self.num_units])

            W_x = tf.get_variable('W_g', [x_size, self.num_units], initializer=orthogonal_initializer())
            b_x = tf.get_variable('W_g', [self.num_units])

            gate_data = tf.concat([x, h, context], 1)
            gate_hidden = tf.matmul(gate_data, W_g) + b_g
            i, f, o = tf.split(1, 3, gate_hidden)

            hidden = tf.matmul(x. W_x) + b_x

            new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(hidden)
            new_h = tf.tanh(new_c) * tf.sigmoid(o)

            return new_h, (new_c, new_h)


class RNN(RNNCell):
    def __init__(self, num_units):
        self.num_units = num_units

    @property
    def state_size(self):
        return (self.num_units, self.num_units)

    @property
    def output_size(self):
        return self.num_units

    def __call__(self, x, state, scope=None):
        with tf.variable_scope(scope or type(self).__name__):
            h = state

            # Keep W_xh and W_hh separate here as well to reuse initialization methods
            x_size = x.get_shape().as_list()[1]
            W_xh = tf.get_variable('W_xh', [x_size, self.num_units])
            W_hh = tf.get_variable('W_hh', [self.num_units, self.num_units])
            bias = tf.get_variable('bias', [self.num_units])

            hidden = tf.matmul(x, W_xh) + bias + tf.matmul(h, W_hh)
            new_h = tf.tanh(hidden)

            return new_h, (new_h)


def bn_lstm_identity_initializer(scale):
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        '''Ugly cause LSTM params calculated in one matrix multiply'''
        size = shape[0]
        # gate (j) is identity
        t = np.zeros(shape)
        t[:, size:size * 2] = np.identity(size) * scale
        t[:, :size] = orthogonal([size, size])
        t[:, size * 2:size * 3] = orthogonal([size, size])
        t[:, size * 3:] = orthogonal([size, size])
        return tf.constant(t, dtype)

    return _initializer
