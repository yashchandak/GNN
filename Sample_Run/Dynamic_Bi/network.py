import tensorflow as tf
import numpy as np
from collections import Counter
import math
from tensorflow.python.ops.seq2seq import sequence_loss
from tensorflow.python.ops.rnn_cell import RNNCell
#from BNlstm import BNLSTMCell
#from tf.nn.rnn_cell import GRUCell, DropoutWrapper, MultiRNNCell

class Network(object):
    
    def __init__(self,config):
        self.config = config
        self.initial_state = tf.zeros([self.config.batch_size, self.config.mRNN._hidden_size])
        self.global_step = tf.Variable(0,name="global_step",trainable=False) #Epoch

    def weight_variable(self,name, shape):
      initial = tf.truncated_normal(shape,stddev=1.0 / shape[0])
      return tf.Variable(initial, name=name )

    def bias_variable(self,shape):
      initial = tf.constant(0.1, shape=shape)
      return tf.Variable(initial)

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            tf.summary.histogram(name, var)

    def embedding(self, inputs):
        """Add embedding layer.
        Returns:
          inputs: List of length num_steps, each of whose elements should be
                  a tensor of shape (batch_size, embed_size).
        """
        
        embedding = tf.get_variable( 'Embedding',[self.config.data_sets._len_vocab, self.config.mRNN._embed_size], trainable=True)
        inputs = tf.nn.embedding_lookup(embedding, inputs)
        inputs = [tf.squeeze(x, [1]) for x in tf.split(1, self.config.num_steps, inputs)]
        return inputs

    def projection(self, rnn_outputs):
        """Adds a projection layer.

        The projection layer transforms the hidden representation to a distribution
        over the vocabulary.

        Hint: Here are the dimensions of the variables you will need to
              create 
              
              U:   (hidden_size, len(vocab))
              b_2: (len(vocab),)

        Args:
          rnn_outputs: List of length num_steps, each of whose elements should be
                       a tensor of shape (batch_size, embed_size).
        Returns:
          outputs: List of length num_steps, each a tensor of shape
                   (batch_size, len(vocab)
        """
        
        with tf.variable_scope('Projection'):
            U = tf.get_variable('Matrix', [self.config.mRNN._hidden_size*2, self.config.data_sets._len_vocab])
            proj_b = tf.get_variable('Bias', [self.config.data_sets._len_vocab])
            outputs = [tf.matmul(o, U) + proj_b for o in rnn_outputs]

            with tf.variable_scope('label'):
                U2 = tf.get_variable('Matrix_label', [self.config.mRNN._hidden_size*2, self.config.data_sets._len_labels])
                proj_b2 = tf.get_variable('Bias_label', [self.config.data_sets._len_labels])
                outputs_labels = [tf.matmul(o, U2) + proj_b2 for o in rnn_outputs]

            self.variable_summaries(U, 'Node_Projection_Matrix')
            self.variable_summaries(U2, 'Label_Projection_Matrix')
            
        return [outputs, outputs_labels]


    def predict2(self,inputs,keep_prob):
        """Build the model up to where it may be used for inference.
        """
        hidden_size = self.config.mRNN._hidden_size
        batch_size = self.config.batch_size
        embed_size = self.config.mRNN._embed_size

        if keep_prob == None:
            keep_prob = 1

        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x,keep_prob) for x in inputs]
                
        with tf.variable_scope('RNN') as scope:
            state = self.initial_state
            RNN_H = tf.get_variable('HMatrix',[hidden_size,hidden_size])
            RNN_I = tf.get_variable('IMatrix', [embed_size,hidden_size])
            RNN_b = tf.get_variable('B',[hidden_size])

            self.variable_summaries(RNN_H, 'HMatrix')
            self.variable_summaries(RNN_I, 'IMatrix')
            self.variable_summaries(RNN_b, 'Bias')
            
        with tf.variable_scope('RNN',reuse=True):
            rnn_outputs = []
            for tstep, current_input in enumerate(inputs):
                RNN_H = tf.get_variable('HMatrix',[hidden_size,hidden_size])
                RNN_I = tf.get_variable('IMatrix', [embed_size,hidden_size])
                RNN_b = tf.get_variable('B',[hidden_size])
                state = tf.nn.tanh(tf.matmul(state,RNN_H) + tf.matmul(current_input,RNN_I) + RNN_b)
                rnn_outputs.append(state)
		#How to pass state info for subsequent sentences
            self.final_state = rnn_outputs[-1]
    
        with tf.variable_scope('RNNDropout'):
            rnn_outputs = [tf.nn.dropout(x,keep_prob) for x in rnn_outputs]

        return rnn_outputs

    def predict3(self, inputs, keep_prob):

        hidden_size = self.config.mRNN._hidden_size
        num_layers  = self.config.mRNN._layers

        if keep_prob == None:
            keep_prob = 1

        with tf.variable_scope('InputDropout'):
            inputs = tf.pack([tf.nn.dropout(x,keep_prob) for x in inputs])
            
        with tf.variable_scope('GRU'):
            cell = tf.nn.rnn_cell.GRUCell(hidden_size)
            #cell = BNLSTMCell(hidden_size)
            cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob = keep_prob)
            cell = tf.nn.rnn_cell.MultiRNNCell([cell] * num_layers)

            outputs, self.final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, time_major = True)

        outputs = tf.unpack(outputs,axis=0)
        
        with tf.variable_scope('RNNDropout'):
            outputs = [tf.nn.dropout(x,keep_prob) for x in outputs]

        return outputs
        

    def predict(self, inputs, keep_prob, seq_len):

        class MyCell(RNNCell):
            #Define new kind of RNN cell
            def __init__(self, num_units):
                self.num_units = num_units

            @property
            def state_size(self):
                return self.num_units

            @property
            def output_size(self):
                return self.num_units

            def __call__(self, x, state, scope=None):
                with tf.variable_scope(scope or type(self).__name__):
                    
                    x_size = x.get_shape().as_list()[1]
                    RNN_H = tf.get_variable('HMatrix',[self.num_units, self.num_units])
                    RNN_I = tf.get_variable('IMatrix', [x_size,self.num_units])
                    RNN_b = tf.get_variable('B',[hidden_size])
                    state = tf.nn.tanh(tf.matmul(state,RNN_H) + tf.matmul(x,RNN_I) + RNN_b)
                    #state = tf.nn.tanh(tf.matmul(state,RNN_H) + x + RNN_b)
                    #state to be passed on should be a tuple
                    return state, state
 

        hidden_size = self.config.mRNN._hidden_size
        num_layers  = self.config.mRNN._layers

        if keep_prob == None:
            keep_prob = 1

        with tf.variable_scope('InputDropout'):
            inputs = [tf.nn.dropout(x,keep_prob) for x in inputs]
            inputs = tf.pack(inputs)
            
        with tf.variable_scope('MyCell'):
            cell_fw = MyCell(hidden_size)
            cell_fw = tf.nn.rnn_cell.DropoutWrapper(cell_fw, output_keep_prob = keep_prob)
            #cell_fw = tf.nn.rnn_cell.MultiRNNCell([cell_fw] * num_layers, state_is_tuple=False)

            cell_bw = MyCell(hidden_size)
            cell_bw = tf.nn.rnn_cell.DropoutWrapper(cell_bw, output_keep_prob = keep_prob)
            #cell_bw = tf.nn.rnn_cell.MultiRNNCell([cell_bw] * num_layers, state_is_tuple=False)

            initialState_fw = self.initial_state#tf.random_normal([self.config.batch_size,hidden_size], stddev=0.1)
            initialState_bw = self.initial_state#tf.random_normal([self.config.batch_size,hidden_size], stddev=0.1)
            
            outputs, output_states  = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, inputs, 
                                                                      initial_state_fw=initialState_fw, initial_state_bw=initialState_bw, 
                                                                      sequence_length=seq_len, dtype=tf.float32, time_major = True)

        self.final_state = output_states
        self.final_state_fw, self.final_state_bw = output_states[0], output_states[1]
        outputs_fw, outputs_bw = outputs[0], outputs[1]  #individual outputs  
        outputs = tf.concat(2, [outputs_fw, outputs_bw]) #concatenated outputs
        outputs = tf.unpack(outputs,axis=0)
        
        with tf.variable_scope('RNNDropout'):
            outputs = [tf.nn.dropout(x,keep_prob) for x in outputs]           

        return outputs

        

    def loss(self, predictions, labels, labels_2, inputs):
        """Calculates the loss from the predictions (logits?) and the labels.
        """
        next_word = labels
        curr_label = tf.cast(labels_2, tf.float32)

        
        prediction_word = predictions[0]
        prediction_label = predictions[1]

        #initialising variables
        cross_entropy_next = tf.constant(0)
        cross_entropy_label = tf.constant(0)
        cross_entropy_label_similarity = tf.constant(0)
        cross_entropy_emb = tf.constant(0)
        
        self.prec_label, self.prec_label_op = tf.constant(1), tf.constant(1)
        self.recall_label, self.recall_label_op =  tf.constant(1), tf.constant(1)
        self.label_sigmoid = tf.constant(0)

        
        if self.config.solver._next_node_loss:
            #Vector to weigh different word losses
            all_ones = [tf.ones([self.config.batch_size * self.config.num_steps])]
            #cross entropy loss for next word prediction
            cross_entropy_next = sequence_loss([prediction_word],[tf.reshape(next_word, [-1])], all_ones, self.config.data_sets._len_vocab)
            tf.add_to_collection('total_loss', cross_entropy_next)
            
        if self.config.solver._curr_label_loss:
            #Get the slice of tensor representing label '0' for all batch.seq
            #'0' label is assigned for <EOS> and the nodes whose labels are not known
            #Valid errors are only those which don't have '0' label
            valid = tf.cast(tf.less(tf.slice(curr_label, [0,0,0], [self.config.num_steps, self.config.batch_size, 1]), tf.constant(0.5)), tf.float32)
            #replicate along 3rd axis
            valid = tf.tile(valid, tf.pack([1,1,tf.shape(curr_label)[2]]))
        
            #Sigmoid activation
            self.label_sigmoid = tf.sigmoid(prediction_label)
            #binary cross entropy for labels
            cross_loss = tf.add(tf.log(1e-10 + self.label_sigmoid)*curr_label,
                                tf.log(1e-10 + (1-self.label_sigmoid))*(1-curr_label))
            #only consider the loss for valid label predictions
            #[TODO] mean of all or mean of only valid ???
            cross_entropy_label = -1*tf.reduce_mean(tf.reduce_sum(cross_loss*valid,2))
            tf.add_to_collection('total_loss', cross_entropy_label)


        if self.config.solver._label_similarity_loss: 
            #Label similarity loss       
            label_sigmoid = tf.sigmoid(pred_label_reshaped)
            part1 = tf.slice(label_sigmoid, [0,0,0], [self.config.num_steps-1, self.config.batch_size, self.config.data_sets._len_labels])
            part2 = tf.slice(label_sigmoid, [1,0,0], [self.config.num_steps-1, self.config.batch_size, self.config.data_sets._len_labels])

            #Exponential weightage -> [r**(n-1), r**(n-2), ... , r**2. r**1]
            label_diffusion = tf.constant([self.config.data_sets._diffusion_rate**i for i in range(self.config.num_steps-1,0,-1)])
            cross_loss_sim = tf.add(tf.log(1e-10 + part1)*part2, tf.log(1e-10 + (1-part1))*(1-part2))
            #prediction is 3 dimensional (seq x batch x label_len), reduce along axis of label_len
            #Sum over each label error -> take mean over the batch -> sum for the sequence
            cross_entropy_label_similarity = tf.reduce_sum(tf.reduce_mean(-tf.reduce_sum(cross_loss_sim, 2),1) * label_diffusion)
            tf.add_to_collection('total_loss', cross_entropy_label_similarity)

            
        if self.config.solver._embedding_loss:
            #embedding similarity loss
            #Matching First input's embeddings with embeddings of other inputs
            #[TODO] reverse feed of input AND reverse diffusion rate
            
            emb_part1 = tf.slice(inputs, [self.config.num_steps-2,0,0], [1, self.config.batch_size, self.config.mRNN._embed_size])
            emb_part2 = tf.slice(inputs, [0,0,0], [self.config.num_steps-1, self.config.batch_size, self.config.mRNN._embed_size])

            #Exponential weightage -> [r**(n-1), r**(n-2), ... , r**2. r**1]
            label_diffusion = tf.constant([self.config.data_sets._diffusion_rate**i for i in range(self.config.num_steps-1,0,-1)])
            #Broadcastive Subtraction
            mse_emb = tf.reduce_mean(tf.square(emb_part2 - emb_part1),2)
            cross_entropy_emb = tf.reduce_sum(tf.reduce_mean(mse_emb,1) * label_diffusion) * self.config.data_sets._emb_factor
            tf.add_to_collection('total_loss', cross_entropy_emb)

        loss = tf.add_n(tf.get_collection('total_loss'))
        
        tf.summary.scalar('next_node_loss', cross_entropy_next)
        tf.summary.scalar('curr_label_loss', cross_entropy_label)
        tf.summary.scalar('label_similarity_loss', cross_entropy_label_similarity )
        tf.summary.scalar('emb_loss', cross_entropy_emb)
        tf.summary.scalar('total_loss', tf.reduce_sum(loss))
        
        return [loss, cross_entropy_next, cross_entropy_label, cross_entropy_label_similarity, cross_entropy_emb]

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
      # Add a scalar summary for the snapshot loss.
      #tf.scalar_summary('loss', loss)
      # Create a variable to track the global step. - iterations
      #global_step = tf.Variable(0, name='global_step', trainable=False)
      #train_op = optimizer.minimize(loss, global_step=global_step)
      # Use the optimizer to apply the gradients that minimize the loss
      # (and also increment the global step counter) as a single training step.
      train_op = optimizer.minimize(loss[0])
      return train_op

