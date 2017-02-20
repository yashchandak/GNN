import tensorflow as tf
import numpy as np
from collections import Counter
import math
from tensorflow.python.ops.seq2seq import sequence_loss
from tensorflow.python.ops.rnn_cell import RNNCell


# from BNlstm import BNLSTMCell
# from tf.nn.rnn_cell import GRUCell, DropoutWrapper, MultiRNNCell

class Network(object):
    def __init__(self, config):
        self.config = config
        self.global_step = tf.Variable(0, name="global_step", trainable=False)  # Epoch

    def weight_variable(self, name, shape):
        initial = tf.truncated_normal(shape, stddev=1.0 / shape[0])
        return tf.Variable(initial, name=name)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            tf.summary.histogram(name, var)

    def projection(self, rnn_outputs):
        """
        Args: rnn_outputs: (batch_size, hidden_size).
        Returns: (batch_size, len_labels )
        """

        with tf.variable_scope('Projection'):
            U = tf.get_variable('Matrix', [self.config.mRNN._hidden_size, self.config.data_sets._len_labels])
            proj_b = tf.get_variable('Bias', [self.config.data_sets._len_labels])
            outputs = tf.matmul(rnn_outputs, U) + proj_b
            # outputs = tf.matmul(rnn_outputs, tf.transpose(self.embedding)) for o in rnn_outputs]

            self.variable_summaries(U, 'Node_Projection_Matrix')
            # self.variable_summaries(U2, 'Label_Projection_Matrix')

        return outputs

    def predict(self, inputs, inputs2, keep_prob_in, keep_prob_out, label_in, state=None):
        # Non-Dynamic Unidirectional RNN
        """
        Args: inputs: (num_steps, batch_size, len_features).
              inputs2:(num_steps, batch_size, len_labels).
              keep_prob_in: float.
              keep_prob_out: float.
              label_in: bool.
              state: (batch_size, hidden_size)

        Returns: (batch_size, hidden_size )
        """

        hidden_size = self.config.mRNN._hidden_size
        feature_size = self.config.data_sets._len_features
        label_size = self.config.data_sets._len_labels
        batch_size = tf.shape(inputs)[1]

        

        # Split along time direction
        inputs = tf.unstack(inputs, axis=0)
        inputs2 = tf.unstack(inputs2, axis=0)

        if state == None:
            state = tf.zeros([batch_size, self.config.mRNN._hidden_size])
        if keep_prob_in == None:
            keep_prob_in = 1
        if keep_prob_out == None:
            keep_prob_out = 1

        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x,keep_prob_in) for x in inputs]

	    inp_cat = tf.pack([tf.concat(1, [inputs[tstep], inputs2[tstep]]) for tstep in range(len(inputs))])
            
        with tf.variable_scope('MyCell'):
            cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            #cell = BNLSTMCell(hidden_size)
            #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = keep_prob)
            #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=False)

            #initState = self.initial_state#tf.random_normal([self.config.batch_size,hidden_size], stddev=0.1)
            outputs, self.final_state = tf.nn.dynamic_rnn(cell, inp_cat, dtype=tf.float32, time_major = True)



        #outputs = tf.unpack(outputs,axis=0) 
        with tf.variable_scope('RNNDropout'):
            self.variable_summaries(self.final_state, 'final_state') #summary wtiter throwing 'noneType' error otherwise
            rnn_outputs = tf.nn.dropout(self.final_state[0], keep_prob_out)

        return rnn_outputs


    def predict2(self, inputs, inputs2, keep_prob_in, keep_prob_out, label_in, state=None):
        # Non-Dynamic Unidirectional RNN
        """
        Args: inputs: (num_steps, batch_size, len_features).
              inputs2:(num_steps, batch_size, len_labels).
              keep_prob_in: float.
              keep_prob_out: float.
              label_in: bool.
              state: (batch_size, hidden_size)

        Returns: (batch_size, hidden_size )
        """

        hidden_size = self.config.mRNN._hidden_size
        feature_size = self.config.data_sets._len_features
        label_size = self.config.data_sets._len_labels
        batch_size = tf.shape(inputs)[1]

        # Split along time direction
        inputs = tf.unstack(inputs, axis=0)
        inputs2 = tf.unstack(inputs2, axis=0)

        if state == None:
            state = tf.zeros([batch_size, self.config.mRNN._hidden_size])
        if keep_prob_in == None:
            keep_prob_in = 1
        if keep_prob_out == None:
            keep_prob_out = 1

        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x, keep_prob_in) for x in inputs]

        with tf.variable_scope('RNN') as scope:

            self.RNN_H = tf.get_variable('HMatrix', initializer=tf.eye(hidden_size))
            self.RNN_I = tf.get_variable('IMatrix', [feature_size, hidden_size])
            self.RNN_LI = tf.get_variable('LIMatrix', [label_size, hidden_size])
            self.RNN_ILI = tf.get_variable('ILIMatrix', [label_size + feature_size, hidden_size])
            self.RNN_b = tf.get_variable('B', [hidden_size])

            self.variable_summaries(self.RNN_H, 'HMatrix')
            self.variable_summaries(self.RNN_I, 'IMatrix')
            self.variable_summaries(self.RNN_LI, 'ILIMatrix')
            self.variable_summaries(self.RNN_b, 'Bias')

            # other things to be tried:
            # a) MLP over feature inputs to reduce dimension
            # b) deeper net on label and feature concatenation

            # Concat feature and estimated label inputs
            if label_in is not None:
                for tstep in range(len(inputs)):
                    state = tf.nn.tanh(tf.matmul(state, self.RNN_H) + self.RNN_b +
                                       tf.matmul(tf.concat(1, [inputs[tstep], inputs2[tstep]]), self.RNN_ILI))

            # Treat features and estimated labels separately
            #if label_in is not None:
            #    for tstep in range(len(inputs)):
            #        state = tf.nn.tanh(tf.matmul(state,self.RNN_H) +
            #                           tf.matmul(inputs[tstep] ,self.RNN_I) +
            #                           tf.matmul(inputs2[tstep],self.RNN_LI + self.RNN_b))

                    # Exclude the input label information for the final step prediction
                    # state = tf.nn.tanh(tf.matmul(state,self.RNN_H) +
                    #                   tf.matmul(inputs[-1],self.RNN_I) +
                    #                   self.RNN_b)

            else:
                for tstep, current_input in enumerate(inputs):
                    state = tf.nn.tanh(tf.matmul(state, self.RNN_H) + tf.matmul(current_input, self.RNN_I) + self.RNN_b)
                    # state = tf.matmul(state,RNN_H) + current_input

            self.final_state = state

        with tf.variable_scope('RNNDropout'):
            rnn_outputs = tf.nn.dropout(self.final_state, keep_prob_out)

        return rnn_outputs

    def loss(self, predictions, labels):
        """
         Args: predictions: (batch_size, len_labels)
               labels: (batch_size, len_labels).

         Returns: (batch_size, len_labels )
         """  # initialising variables
        cross_entropy_label = tf.constant(0)
        self.label_sigmoid = tf.constant(0)

        if self.config.solver._curr_label_loss:
            # Sigmoid activation
            # self.label_sigmoid = tf.sigmoid(predictions)
            self.label_sigmoid = tf.nn.softmax(predictions)
            cross_entropy_label = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(self.label_sigmoid + 1e-10), 1))
            # binary cross entropy for labels
            #cross_loss = tf.add(tf.log(1e-10 + self.label_sigmoid)*labels,
            #                   tf.log(1e-10 + (1-self.label_sigmoid))*(1-labels))
            #cross_entropy_label = -1*tf.reduce_mean(tf.reduce_sum(cross_loss,1))

            tf.add_to_collection('total_loss', cross_entropy_label)

        if self.config.solver._L2loss:
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) *0
            tf.add_to_collection('total_loss', lossL2)

        loss = tf.add_n(tf.get_collection('total_loss'))
        #grads = tf.gradients(loss, [self.final_state, self.final_state, self.final_state])
        tf.summary.scalar('curr_label_loss', cross_entropy_label)
        tf.summary.scalar('total_loss', tf.reduce_sum(loss))

        return [loss, cross_entropy_label]#, grads]


    def training(self, loss, optimizer):
        """Sets up the training Ops.
        Creates a summarizer to track the loss over time in TensorBoard.
        Creates an optimizer and applies the gradients to all trainable variables.
        The Op returned by this function is what must be passed to the
        `sess.run()` call to cause the model to train.
        Args:
          loss: Loss tensor, from loss().
          learning_rate: The learning rate to use for gradient descent.
        Returns:
          train_op: The Op for training.
        """
        train_op = optimizer.minimize(loss[0])
        return train_op
