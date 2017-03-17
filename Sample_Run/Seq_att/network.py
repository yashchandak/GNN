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

            self.variable_summaries(U, 'Node_Projection_Matrix')

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

        class MyLSTMCell(RNNCell):
            '''Vanilla LSTM implemented with same initializations as BN-LSTM'''

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
                    c, h = state

                    # Keep W_xh and W_hh separate here as well to reuse initialization methods
                    x_size = x.get_shape().as_list()[1]
                    W_xh = tf.get_variable('W_xh',
                                           [x_size, 4 * self.num_units],
                                           initializer=orthogonal_initializer())
                    W_hh = tf.get_variable('W_hh',
                                           [self.num_units, 4 * self.num_units],
                                           initializer=bn_lstm_identity_initializer(0.95))
                    bias = tf.get_variable('bias', [4 * self.num_units])

                    # hidden = tf.matmul(x, W_xh) + tf.matmul(h, W_hh) + bias
                    # improve speed by concat.
                    concat = tf.concat(1, [x, h])
                    W_both = tf.concat(0, [W_xh, W_hh])
                    hidden = tf.matmul(concat, W_both) + bias

                    i, j, f, o = tf.split(1, 4, hidden)

                    new_c = c * tf.sigmoid(f) + tf.sigmoid(i) * tf.tanh(j)
                    new_h = tf.tanh(new_c) * tf.sigmoid(o)

                    return new_h, (new_c, new_h)

        hidden_size = self.config.mRNN._hidden_size
        feature_size = self.config.data_sets._len_features
        label_size = self.config.data_sets._len_labels
        batch_size = tf.shape(inputs)[1]
        max_len = self.config.num_steps

        if self.config.data_sets.reduced_dims:
            with tf.variable_scope('Reduce_Dim') as scope:
                W_ii = tf.get_variable('W_ii', [feature_size, self.config.data_sets.reduced_dims])
                inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, feature_size]), W_ii), [max_len,-1,self.config.data_sets.reduced_dims])
                scope.reuse_variables()

        # Split along time direction
        inputs = tf.unstack(inputs, axis=0)
        inputs2 = tf.unstack(inputs2, axis=0)

        if state == None:
            #initState = self.initial_state#tf.random_normal([self.config.batch_size,hidden_size], stddev=0.1)
            state = (tf.zeros([batch_size, self.config.mRNN._hidden_size]),
                     tf.zeros([batch_size, self.config.mRNN._hidden_size]))
        if keep_prob_in == None:
            keep_prob_in = 1
        if keep_prob_out == None:
            keep_prob_out = 1

        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x,keep_prob_in) for x in inputs]

	    #inp_cat = tf.pack([tf.concat(1, [inputs[tstep], inputs2[tstep]]) for tstep in range(len(inputs))])
	    inp_cat = [tf.concat(1, [inputs[tstep], inputs2[tstep]]) for tstep in range(len(inputs))]

        with tf.variable_scope('MyCell'):
            #cell = tf.nn.rnn_cell.GRUCell(hidden_size)
            #cell = tf.nn.rnn_cell.LSTMCell(hidden_size)
            cell = MyLSTMCell(hidden_size)
            #cell = BNLSTMCell(hidden_size)
            #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = keep_prob)
            #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers, state_is_tuple=False)

            #outputs, self.final_state = tf.nn.dynamic_rnn(cell, inp_cat, dtype=tf.float32, time_major = True)
            with tf.variable_scope('Loop') as scope:
                rnn_outputs = []
                l = len(inp_cat)
                d = max(0, l - self.config.max_depth) #clip maximum depth to be traversed
                for tstep in range(d, l):
                    outs, state = cell.__call__(inp_cat[tstep], state=state)
                    rnn_outputs.append(outs)
                    scope.reuse_variables()

                # How to pass state info for subsequent sentences
                self.final_state = state

        context = inputs[-1] #Treat the attribute of node-of-interest as context for attention
        att_state = self.attention(rnn_outputs, context)

        #outputs = tf.unpack(outputs,axis=0)
        with tf.variable_scope('RNNDropout'):
            self.variable_summaries(self.final_state, 'final_state') #summary wtiter throwing 'noneType' error otherwise
            #rnn_outputs = tf.nn.dropout(self.final_state[0], keep_prob_out)
            rnn_outputs = tf.nn.dropout(att_state, keep_prob_out)

        return rnn_outputs


    def attention2(self, states, context, attn_size=None):
        states = tf.pack(states) #convert from list to tensor
        states = tf.transpose(states, [1,0,2]) # [Num_step, Batch, state_size] -> [Batch, Num_step, state_size]

        batch_size, num_step, state_size = states.get_shape().as_list()
        context_size = context.get_shape().as_list()[-1]
        attn_length = num_step #length of attention vector = Num_step

        hidden = tf.reshape(states,[-1, attn_length, 1, state_size]) # [Batch, Num_step, 1, state_size]

        with tf.variable_scope('Attention') as scope:
            # filter
            score_weight = tf.get_variable("W_attention_softmax", [context_size, state_size]) #[context_size, state_size]
            k = tf.matmul(context, score_weight) #[Batch, Context_size] * [context_size, state_size] -> [Batch, state_size]
            k = tf.reshape(k, [-1, 1, state_size, 1])  # Reshape into 4D: [Batch, 1, state_size, 1]
            attn_features = tf.nn.conv2d(hidden, k, [1,1,1,1], "SAME") # [Batch, Num_step, 1, state_size] * [1, 1, state_size, 1] = [Batch, Num_step, 1, 1]

            # Calculating alpha
            s = tf.reshape(attn_features, [-1, attn_length]) # [Batch, Num_step, 1, 1] -> [Batch, Num_step]
            #a = s # [Batch, Num_step]
            #a = tf.nn.sigmoid(s) # [Batch, Num_step]
            a = tf.nn.softmax(s) # [Batch, Num_step]

            # Calculate context c
            c = tf.reduce_sum(tf.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2]) #[Batch, Num_step, 1, 1]* [Batch, Num_step, 1, state_size] -> [Batch, state_size]
            scope.reuse_variables()

        return c

    def attention(self, states, context, attn_size=None):
        states = tf.pack(states) #convert from list to tensor
        states = tf.transpose(states, [1,0,2]) # [Num_step, Batch, state_size] -> [Batch, Num_step, state_size]

        batch_size, num_step, state_size = states.get_shape().as_list()
        if attn_size == None:  # size of the intermediate attention representation
            attn_size = state_size  # by default A = state_size
        attn_length = num_step #length of attention vector = Num_step

        hidden = tf.reshape(states,[-1, attn_length, 1, state_size]) # [Batch, Num_step, 1, state_size]
        # filter
        k = tf.get_variable("AttnW",[1, 1, state_size, attn_size]) # [1, 1, state_size, A]
        attn_features = tf.nn.conv2d(hidden, k, [1,1,1,1], "SAME") # [Batch, Num_step, 1, state_size] * [1, 1, state_size, A] = [Batch, Num_step, 1, A]
        attention_softmax_weights = tf.get_variable("W_attention_softmax", [attn_size]) # [A]

        y = tf.nn.rnn_cell._linear(args = context, output_size = attn_size, bias = True) # W*C + b : [Batch, context_size] -> [Batch, A]
        y = tf.reshape(y, [-1, 1, 1, attn_size])  # Reshape into 4D: [Batch, 1, 1, A]

        # Calculating alpha
        #s = tf.reduce_sum(attention_softmax_weights * tf.nn.tanh(attn_features + y), [2, 3]) # [A]*[Batch, Num_step, 1, A] -> [Batch, Num_step]
        s = tf.reduce_sum(attention_softmax_weights * tf.nn.tanh(attn_features + y), [2, 3]) # [A]*[Batch, Num_step, 1, A] -> [Batch, Num_step]
        #a = s # [Batch, Num_step]
        #a = tf.nn.sigmoid(s) # [Batch, Num_step]
        a = tf.nn.softmax(s) # [Batch, Num_step]

        # Calculate context c
        c = tf.reduce_sum(tf.reshape(a, [-1, attn_length, 1, 1]) * hidden, [1, 2]) #[Batch, Num_step, 1, 1]* [Batch, Num_step, 1, state_size] -> [Batch, state_size]

        return c



    def loss(self, predictions, labels):
        """
         Args: predictions: (batch_size, len_labels)
               labels: (batch_size, len_labels).

         Returns: (batch_size, len_labels )
         """  # initialising variables
        cross_entropy_label = tf.constant(0)
        self.label_preds = tf.constant(0)

        if self.config.solver._curr_label_loss:
            if self.config.data_sets._multi_label:
                # Sigmoid activation
                self.label_preds = tf.nn.sigmoid(predictions)
                # binary cross entropy for labels
                cross_loss = tf.add(tf.log(1e-10 + self.label_preds)*labels,
                                   tf.log(1e-10 + (1-self.label_preds))*(1-labels))
                cross_entropy_label = -1*tf.reduce_mean(tf.reduce_sum(cross_loss,1))

            else:
                self.label_preds = tf.nn.softmax(predictions)
                cross_entropy_label = tf.reduce_mean(-tf.reduce_sum(labels * tf.log(self.label_preds + 1e-10), 1))

            tf.add_to_collection('total_loss', cross_entropy_label)

        if self.config.solver._L2loss:
            vars = tf.trainable_variables()
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars]) * self.config.solver._L2loss
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


def orthogonal(shape):
    flat_shape = (shape[0], np.prod(shape[1:]))
    a = np.random.normal(0.0, 1.0, flat_shape)
    u, _, v = np.linalg.svd(a, full_matrices=False)
    q = u if u.shape == flat_shape else v
    return q.reshape(shape)

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

def orthogonal_initializer():
    def _initializer(shape, dtype=tf.float32, partition_info=None):
        return tf.constant(orthogonal(shape), dtype)
    return _initializer