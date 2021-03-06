import tensorflow as tf
import numpy as np
from cells import RNN, MyLSTMCell, LSTMgated
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

    def projection(self, attn_outputs, x_feats, x_labels):
        """
        Args: rnn_outputs: (batch_size, hidden_size).
        Returns: (batch_size, len_labels )
        """
        _, inp_dim = attn_outputs.get_shape().as_list()
        NOI_feats = tf.concat(0, [x_feats[-1][0], x_labels[-1][0]])
        with tf.variable_scope('Projection'):
            U = tf.get_variable('Matrix', [inp_dim, self.config.data_sets._len_labels])
            U2 = tf.get_variable('Matrix2', [self.config.data_sets._len_features + self.config.data_sets._len_labels,
                                             self.config.data_sets._len_labels])
            proj_b = tf.get_variable('Bias', [self.config.data_sets._len_labels])
            outputs = tf.matmul(attn_outputs, U) + tf.matmul([NOI_feats], U2) + proj_b

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
        hidden_size = self.config.mRNN._hidden_size
        feature_size = self.config.data_sets._len_features
        label_size = self.config.data_sets._len_labels
        batch_size = tf.shape(inputs)[1]
        max_len = self.config.num_steps

        if self.config.data_sets.reduced_dims:
            with tf.variable_scope('Reduce_Dim') as scope:
                W_ii = tf.get_variable('W_ii', [feature_size, self.config.data_sets.reduced_dims])
                W_iib = tf.get_variable('W_ii_bias', [self.config.data_sets.reduced_dims])
                inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, feature_size]), W_ii) + W_iib, [max_len,-1,self.config.data_sets.reduced_dims])
                #inputs = tf.reshape(tf.matmul(tf.reshape(inputs, [-1, feature_size]), W_ii), [max_len,-1,self.config.data_sets.reduced_dims])
                scope.reuse_variables()


        # Split along time direction
        inputs = tf.unstack(inputs, axis=0)
        inputs2 = tf.unstack(inputs2, axis=0)
        context = inputs[-1]

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
            if self.config.mRNN.cell == 'GRU':
                cell_type = tf.nn.rnn_cell.GRUCell
                state = state[0]
            if self.config.mRNN.cell == 'RNN':
                cell_type = RNN
                state = state[0]
            elif self.config.mRNN.cell == 'LSTM': cell_type = tf.nn.rnn_cell.LSTMCell
            elif self.config.mRNN.cell == 'myLSTM': cell_type = MyLSTMCell
            elif self.config.mRNN.cell == 'LSTMgated': cell_type = LSTMgated
            else: raise ValueError('Invalid Cell type')

            cell = cell_type(hidden_size)
            cell2 = tf.nn.rnn_cell.BasicRNNCell(hidden_size)
            # cell2 = cell_type(hidden_size)
            #cell = BNLSTMCell(hidden_size)
            #cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = .5)
            #cell = tf.nn.rnn_cell.MultiRNNCell([cell] * 2, state_is_tuple=True)

            #outputs, self.final_state = tf.nn.dynamic_rnn(cell, inp_cat, dtype=tf.float32, time_major = True)
            with tf.variable_scope('Loop') as scope:
                rnn_outputs = []
                l = len(inp_cat)
                d = max(0, l - self.config.max_depth) #clip maximum depth to be traversed
                for tstep in range(d, l-1):
                    outs, state = cell.__call__(inp_cat[tstep], state=state)#, context=context)
                    if self.config.mRNN.concat:
                        outs = tf.concat(1, [outs, inp_cat[tstep]])
                    rnn_outputs.append(outs)
                    scope.reuse_variables()

            # with tf.variable_scope('Loop2') as scope:
            #    c, h = state
            #    outs, state = cell2.__call__(inp_cat[tstep+1], state=c)
            #    rnn_outputs.append(outs)
            #    scope.reuse_variables()

            self.final_state = state

        #context = inputs[-1] #Treat the attribute of node-of-interest as context for attention
        self.attn_vals = tf.constant(0)
        if self.config.mRNN.attention == 0: att_state = self.final_state[0]
        #elif self.config.mRNN.attention == 1: att_state = self.attention1(rnn_outputs, context)
        #elif self.config.mRNN.attention == 2: att_state = self.attention2(rnn_outputs, context)
        else: raise ValueError('Invlaid attention module')

        #outputs = tf.unpack(outputs,axis=0)
        with tf.variable_scope('RNNDropout'):
            self.variable_summaries(self.final_state, 'final_state') #summary wtiter throwing 'noneType' error otherwise
            rnn_outputs = tf.nn.dropout(att_state, keep_prob_out)

        return rnn_outputs


    def attention1(self, states, context, attn_size=None):
        states = tf.pack(states) #convert from list to tensor
        states = tf.transpose(states, [1,0,2]) # [Num_step, Batch, state_size] -> [Batch, Num_step, state_size]

        batch_size, num_step, state_size = states.get_shape().as_list()
        context_size = context.get_shape().as_list()[-1]
        attn_length = num_step #length of attention vector = Num_step

        hidden = tf.reshape(states,[-1, attn_length, 1, state_size]) # [Batch, Num_step, state_size] -> [Batch, Num_step, 1, state_size]
        hiddenT = tf.transpose(hidden, [1,0,2,3])# [Num_step, Batch, 1, state_size]

        with tf.variable_scope('Attention') as scope:
            # filter
            W = tf.get_variable("W_attention_softmax", [context_size, state_size]) #[context_size, state_size]
            CxW = tf.matmul(context, W) #[Batch, context_size]*[context_size, state_size] -> [Batch, state_size]
            CxW = tf.reshape(CxW, [-1, 1, state_size, 1]) #[Batch, 1, state_size, 1]
            attn_features = tf.nn.conv2d(hiddenT, CxW, [1,1,1,1], "SAME") # [Num_step, Batch, 1, state_size] * [Batch, 1, state_size, 1] = [Num_step, Batch, 1, 1]

            # Calculating alpha
            attn_features = tf.transpose(attn_features, [1,0,2,3]) # [Num_step, Batch, 1, 1] -> [Batch, Num_step, 1, 1]
            s = tf.reshape(attn_features, [-1, attn_length]) # [Batch, Num_step, 1, 1] -> [Batch, Num_step]
            #a = tf.nn.sigmoid(s) # [Batch, Num_step]
            self.attn_vals = tf.nn.softmax(s) # [Batch, Num_step]

            # Calculate context c
            c = tf.reduce_sum(tf.reshape(self.attn_vals, [-1, attn_length, 1, 1]) * hidden, [1, 2]) #[Batch, Num_step, 1, 1]* [Batch, Num_step, 1, state_size] -> [Batch, state_size]
            scope.reuse_variables()

        return c

    def attention(self, states, context, attn_size=None):
        states = tf.pack(states) #convert from list to tensor: [path, state_size]
        context = context[-1][0] #context = last time step's first attribute

        path_size, state_size = states.get_shape().as_list()
        if attn_size == None:  # size of the intermediate attention representation
            attn_size = state_size  # by default A = state_size
        attn_length = path_size #length of attention vector = number of paths
        score_weights = tf.get_variable("ScoreW", [attn_size])  # [A]

        #attention
        k = tf.get_variable("AttnW", [state_size, attn_size])  # [state_size, A]
        attn_features = tf.matmul(states, k) # [path, state_size] * [state_size, A] -> [path,  A]

        y = tf.nn.rnn_cell._linear(args = tf.reshape(context, [1,-1]), output_size = attn_size, bias = True) # W*C + b : [1, context_size] -> [1, A]

        # Calculating alpha
        s = tf.matmul( tf.nn.tanh(attn_features + y), tf.reshape(score_weights, [-1,1])) # [path, A]*[A, 1] -> [path, 1]
        self.attn_vals = tf.nn.softmax(s, dim=0) # [path, 1]

        # Calculate context c
        c = tf.reduce_sum(self.attn_vals * states, [0]) #[path, 1]* [path, state_size] -> [ state_size]

        return tf.reshape(c, [1, -1]) #[1, state_size]



    def loss(self, predictions, labels, wce):
        """
         Args: predictions: (batch_size, len_labels)
               labels: (batch_size, len_labels).

         Returns: (batch_size, len_labels )
         """
        if self.config.data_sets._multi_label:
            # Sigmoid activation
            self.label_preds = tf.nn.sigmoid(predictions)
            # binary cross entropy for labels
            cross_loss = tf.add(tf.log(1e-10 + self.label_preds)*labels,
                               tf.log(1e-10 + (1-self.label_preds))*(1-labels))
            cross_entropy_label = -1*tf.reduce_mean(tf.reduce_sum(wce*cross_loss,1))

        else:
            self.label_preds = tf.nn.softmax(predictions)
            cross_loss = labels * tf.log(self.label_preds + 1e-10)
            cross_entropy_label = tf.reduce_mean(-tf.reduce_sum(wce*cross_loss, 1))

        tf.add_to_collection('total_loss', cross_entropy_label)

        if self.config.solver._L2loss:
            lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.config.solver._L2loss
            tf.add_to_collection('total_loss', lossL2)

        loss = tf.add_n(tf.get_collection('total_loss'))

        tf.summary.scalar('curr_label_loss', cross_entropy_label)
        tf.summary.scalar('total_loss', tf.reduce_sum(loss))

        return loss


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

    def custom_training(self, loss, optimizer, batch_size):

        # gradient accumulation over multiple batches
        # http://stackoverflow.com/questions/42156957/how-to-update-model-parameters-with-accumulated-gradients
        # https://github.com/DrSleep/tensorflow-deeplab-resnet/issues/18#issuecomment-279702843

        tvs = tf.trainable_variables()
        accum_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        reset_op = [v.assign(tf.zeros_like(v)) for v in accum_grads]

        gvs = tf.gradients(loss, tvs) #compute gradients
        # gvs = optimizer.compute_gradients(loss, tvs)

        accum_op = [accum_grads[i].assign_add(gv) for i, gv in enumerate(gvs)] #accumulate computed gradients
        grads = np.asarray(accum_grads) / batch_size #take mean before updating
        update_op = optimizer.apply_gradients(zip(grads,tvs))
        # update_op = optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

        return reset_op, accum_op, update_op