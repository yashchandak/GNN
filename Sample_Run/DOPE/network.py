import numpy as np
import tensorflow as tf
from cells import RNN, myRNN, MyLSTMCell, LSTMgated
from utils import relu_init, tanh_init, zeros, const

class Network(object):

    def __init__(self, config):
        tf.set_random_seed(1234)
        self.config = config
        self.global_step = tf.Variable(0, name="global_step", trainable=False)  # Epoch
        self.cell = None
        self.attn_values = tf.constant(0)

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

    def get_path_data(self, x_attr, x_labels, x_lengths, keep_prob_in, keep_prob_out,  state=None):
        # Non-Dynamic Unidirectional RNN
        """
        Args: x_attr: (num_steps, batch_size, len_features).
              x_labels:(num_steps, batch_size, len_labels).
              keep_prob_in: float.
              keep_prob_out: float.
              length: (num_steps).
              state: (batch_size, hidden_size)

        Returns: (batch_size, hidden_size )
        """
        if keep_prob_in is None:
            keep_prob_in = 1
        if keep_prob_out is None:
            keep_prob_out = 1

        x_dims = self.config.data_sets._len_features
        batch_size = tf.shape(x_attr)[1]
        max_len = self.config.num_steps

        # Weird TF bug, forgets dimensions
        # https://github.com/tensorflow/tensorflow/issues/3102
        x_attr.set_shape([self.config.num_steps, None, self.config.data_sets._len_features])
        x_labels.set_shape([self.config.num_steps, None, self.config.data_sets._len_labels])

        # Select the Node Of Interest (NOI) attributes to form context for attention
        # t_node_id = ids[x_lengths[0]-1][0]
        NOI_x = x_attr[x_lengths[0]-1][0]
        NOI_x.set_shape([self.config.data_sets._len_features, ])

        #Attribute dimensionality reduction : rd_x - reduced x
        if self.config.data_sets.reduced_dims:
            rd_x_dims = self.config.data_sets.reduced_dims
            with tf.variable_scope('Reduce_Dimension') as scope:
                w_rd = tf.get_variable('W_RD', [x_dims, rd_x_dims])
                b_rd = tf.get_variable('b_RD', [rd_x_dims])
                x_attr = tf.nn.relu(tf.matmul(tf.reshape(x_attr, [-1, x_dims]), w_rd) + b_rd)
                x_attr = tf.reshape(x_attr, [self.config.num_steps, batch_size, rd_x_dims])
                scope.reuse_variables()

        with tf.variable_scope('InputDropout'):
            x_attr = tf.nn.dropout(x_attr, keep_prob_in)

        inputs = tf.concat([x_attr, x_labels], 2)

        with tf.variable_scope('MyCell'):
            if self.config.mRNN.cell == 'GRU':
                cell = tf.contrib.rnn.GRUCell(self.config.mRNN._hidden_size)
            elif self.config.mRNN.cell == 'RNN':
                cell = tf.contrib.rnn.BasicRNNCell(self.config.mRNN._hidden_size)
            elif self.config.mRNN.cell == 'LSTM':
                cell = tf.contrib.rnn.LSTMCell(self.config.mRNN._hidden_size)
            elif self.config.mRNN.cell == 'myRNN':
                cell_type = myRNN
                cell = cell_type(self.config.mRNN._hidden_size,
                                 ([self.config.data_sets._len_features, self.config.data_sets._len_labels]))
                self.cell = cell
            elif self.config.mRNN.cell == 'myLSTM':
                cell_type = MyLSTMCell
                cell = cell_type(self.config.mRNN._hidden_size,
                                      [self.config.data_sets._len_features, self.config.data_sets._len_labels])
                self.cell = cell
            elif self.config.mRNN.cell == 'LSTMgated':
                cell_type = LSTMgated
                cell = cell_type(self.config.mRNN._hidden_size)
            else:
                raise ValueError('Invalid Cell type')

            _, self.final_state = tf.nn.dynamic_rnn(cell, inputs, sequence_length=x_lengths-1, dtype=tf.float32,
                                                    time_major=True)
        #Saving the last time step alone
        rnn_outputs = self.final_state[0]

        with tf.variable_scope('RNNDropout'):
            self.variable_summaries(self.final_state, 'final_state') #summary writer throwing 'noneType' error otherwise
            rnn_outputs = tf.nn.dropout(rnn_outputs, keep_prob_out)

        return rnn_outputs, NOI_x, batch_size

    def get_NOI_data(self, x_attr, keep_prob_in):
        if self.cell is not None :
            print('Sharing LSTM''s candidate value projection''s weights ')
            W_x, b_x = self.cell.candidate_weights_bias()
        else:
            W_x = tf.get_variable('W_x', [self.config.data_sets._len_features, self.config.mRNN._hidden_size])
            b_x = tf.get_variable('b_x', [self.config.mRNN._hidden_size])

        x_attr = tf.nn.dropout(tf.expand_dims(x_attr, 0), keep_prob_in)
        #x_hidden = tf.tanh(tf.matmul(x_attr, W_x) + b_x)
        x_hidden = tf.nn.relu(tf.matmul(x_attr, W_x) + b_x)

        return x_hidden

    def attentive_ensemble(self, x_data, neigh_data, attn_size=None):

        with tf.variable_scope('Attentive_Ensemble'):
            states = tf.stack(neigh_data)  # convert from list to tensor: [path, state_size]
            n_path, state_size = states.get_shape().as_list()
            context_size = x_data.get_shape().as_list()[1]
            context = x_data

            if self.config.mRNN.attention == 0: #Don't remove == 0
                c = tf.reduce_mean(states, axis=0)
            else:
                print('Attention model')
                if attn_size is None:  # size of the intermediate attention representation
                    attn_size = state_size  # by default A = state_size
                attn_length = n_path  # length of attention vector = number of paths

                # Attention Mechanism
                score_weights = tf.get_variable("ScoreW", [attn_size, 1])  # [A, 1]

                k = tf.get_variable("AttnW", [state_size, attn_size])  # [state_size, A]
                attn_features = tf.matmul(states, k)  # [path, state_size] * [state_size, A] -> [path,  A]

                W = tf.get_variable("linearW", [context_size, attn_size])
                b = tf.get_variable("linearB", [attn_size])
                y = tf.matmul(context, W) + b  # W*C + b : [1, context_size] -> [1, A]
                # y = tf.nn.rnn_cell._linear(args=tf.reshape(context, [1,-1]), output_size=attn_size, bias = True)

                # Calculating alpha
                s = tf.matmul(tf.nn.tanh(attn_features + y), score_weights)  # [path, A]*[A, 1] -> [path, 1]
                self.attn_values = tf.nn.softmax(s, dim=0)  # [path, 1]

                # Calculate context c
                c = tf.reduce_sum(self.attn_values * states, [0])  # [path, 1]* [path, state_size] -> [ state_size]

            return tf.reshape(c, [1, -1])  # [1, state_size]

    def predict(self, x_data, paths_data, keep_prob_out, concat=False, deep_project=False, attentive_combine=False):

        hid_dim = self.config.mRNN._hidden_size
        if paths_data is not None:
            if not attentive_combine:
                if concat:
                    data = tf.concat([x_data, paths_data], axis=1)
                    hid_dim = 2 * self.config.mRNN._hidden_size
                else:
                    data = x_data + paths_data
        else:
            data = x_data

        if deep_project:
            W_h_o = tf.get_variable('Wh_O', [hid_dim, self.config.mRNN._hidden_size])
            b_h_o = tf.get_variable('bh_O',  [self.config.mRNN._hidden_size])
            data = tf.tanh(tf.matmul(data, W_h_o) + b_h_o)
            data = tf.nn.dropout(data, keep_prob_out)
            hid_dim = self.config.mRNN._hidden_size

        W_o = tf.get_variable('W_o', [hid_dim, self.config.data_sets._len_labels])
        b_o = tf.get_variable('b_o', [self.config.data_sets._len_labels])

        if not self.config.data_sets._multi_label:
            predictions = tf.matmul(data, W_o) + b_o
            predictions = tf.nn.softmax(predictions)
        else:
            predictions = tf.sigmoid(tf.matmul(data, W_o) + b_o)

        return predictions

    def consensus_loss(self, predictions, pred_mean):
        pred_mean = tf.reduce_mean(predictions)
        cross_loss = -1*tf.reduce_mean(tf.multiply(pred_mean, tf.log(1e-10 + predictions)))
        return cross_loss

    def loss(self, predictions, labels, wce):

        if self.config.data_sets._multi_label:
            cross_loss = tf.add(tf.log(1e-10 + predictions) * labels,
                                tf.log(1e-10 + (1 - predictions)) * (1 - labels))
            cross_entropy_label = -1 * tf.reduce_mean(tf.reduce_sum(wce * cross_loss, 1))
        else:
            cross_loss = labels * tf.log(predictions + 1e-10)
            cross_entropy_label = tf.reduce_mean(-tf.reduce_sum(wce * cross_loss, 1))
        tf.summary.scalar('curr_label_loss', cross_entropy_label)
        tf.add_to_collection('total_loss', cross_entropy_label)

        if self.config.solver._L2loss:
            L2_loss = tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()]) * self.config.solver._L2loss
            tf.add_to_collection('total_loss', L2_loss)
            tf.summary.scalar('L2_loss', L2_loss)

        loss = tf.add_n(tf.get_collection('total_loss'))
        tf.summary.scalar('total_loss', tf.reduce_sum(loss))

        return loss

    def training(self, loss, optimizer):
        train_op = optimizer.minimize(loss[0])
        return train_op

    def custom_training(self, loss, optimizer, batch_size):

        # gradient accumulation over multiple batches
        # http://stackoverflow.com/questions/42156957/how-to-update-model-parameters-with-accumulated-gradients
        # https://github.com/DrSleep/tensorflow-deeplab-resnet/issues/18#issuecomment-279702843
        #batch_size = tf.Print(batch_size, [batch_size], message="Batch size: ")

        tvs = tf.trainable_variables()
        accum_grads = [tf.Variable(tf.zeros_like(tv.initialized_value()), trainable=False) for tv in tvs]
        reset_op = [v.assign(tf.zeros_like(v)) for v in accum_grads]

        gvs = tf.gradients(loss, tvs)  # compute gradients
        # gvs = optimizer.compute_gradients(loss, tvs)
        accum_op = [accum_grads[i].assign_add(gv) for i, gv in enumerate(gvs)]  # accumulate computed gradients

        normalized_grads = [var/batch_size for var in accum_grads]
        #grads = np.asarray(accum_grads)
        #grads = np.asarray(accum_grads) / batch_size  # take mean before updating
        update_op = optimizer.apply_gradients(zip(normalized_grads, tvs))
        # update_op = optimizer.apply_gradients([(accum_vars[i], gv[1]) for i, gv in enumerate(gvs)])

        return reset_op, accum_op, update_op





