from __future__ import print_function
import os.path
import time, math, sys
from copy import deepcopy
import scipy.sparse as sps
from scipy.io import loadmat
import numpy as np
from sklearn.preprocessing import normalize

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector

from blogDWdata import DataSet
import network as architecture
import Config as conf
import Eval_Calculate_Performance as perf

#import Eval_MLP as NN
import Eval_linear as liblinear
import Eval_Config

cfg = conf.Config() 
class RNNLM_v1(object):
    def __init__(self, config):
        self.config = config
        # Generate placeholders for the images and labels.
        self.load_data()
        self.add_placeholders()
        # Build model
        self.arch        = self.add_network(config)

        self.rnn_outputs = self.arch.predict(self.data_placeholder,self.keep_prob)
        self.outputs     = self.arch.projection(self.rnn_outputs)
        self.loss        = self.arch.loss(self.outputs, self.label_placeholder, self.data_placeholder)

        self.optimizer   = self.config.solver._parameters['optimizer']
        self.train       = self.arch.training(self.loss,self.optimizer)

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        self.summary = tf.summary.merge_all()
        self.step_incr_op = self.arch.global_step.assign(self.arch.global_step+1)
        self.init = tf.global_variables_initializer()
        
    #[!!!]CHANGE THIS
    def predict_results(self,sess, all_labels, return_labels = False):
        labels_orig, all_data, labels_pred = [], [], []
        diff = 0
        
        for k,v in all_labels.items():
            labels_orig.append(v)
            all_data.append([k])

        for idx in range(0, len(all_data), self.config.batch_size):
            #Replicate data on 2nd axis to meet the dimensions of data placeholder
            #But since dynamic RNNs are used, only lengths of 'seq_length' are evaluated :)
            data = all_data[idx: idx+self.config.batch_size]
            diff = self.config.batch_size - len(data)

            #0 pad the data for final batch if total length is less than batch_size
            if diff !=  0:
                data.extend([[0]]*diff)
                
            data = np.tile(data, (1, self.config.num_steps))
            feed_dict = {self.data_placeholder: data, self.keep_prob: 1, self.arch.initial_state: self.arch.initial_state.eval(), self.seq_len: [1]*len(data)}
            labels_pred.extend(sess.run(self.arch.label_sigmoid, feed_dict=feed_dict)[0])

        labels_pred = labels_pred[: -diff] #Ignore the labels predicted for 0 paddings
        if return_labels:
            return labels_pred
        else:
            return perf.evaluate(labels_pred, labels_orig, 0)

    def load_data(self):
        # Get the 'encoded data'
        self.dataset =  DataSet(self.config)
        debug = self.config.debug
        if debug:
            print('##############--------- Debug mode [NOT IMPLEMENTED] ')
            num_debug = (self.config.num_steps+1)*128
            #self.data_sets.train._x = self.data_sets.train._x[:num_debug]
            #self.data_sets.validation._x  = self.data_sets.validation._x[:num_debug]
            #self.data_sets.test_x  = self.data_sets.test_x[:num_debug]
        
        self.config.data_sets._len_vocab = len(self.dataset.features)
        l = len(list(self.data_sets.train.labels.values())[0])
        self.config.data_sets._len_labels = l
        self.config.data_sets._len_features = self.dataset.features.shape[1]

        print('--------- Project Path: '+self.config.codebase_root_path+self.config.project_name)
        print('--------- Vocabulary Length: '+str(self.config.data_sets._len_vocab))
        print('--------- Label Length: '+str(self.config.data_sets._len_labels))
        print('--------- No. of Labelled nodes: ' + str(len(self.dataset.labels.keys())))

    def add_placeholders(self):
        self.data_placeholder    = tf.placeholder(tf.float32, name='Input')
        self.label_placeholder   = tf.placeholder(tf.float32,name='Target')
        self.keep_prob           = tf.placeholder(tf.float32, name='keep_prob')

    def create_feed_dict(self, input_batch, label_batch):
        feed_dict = {
            self.data_placeholder: input_batch,
            self.label_placeholder: label_batch
        }
        return feed_dict

    def add_network(self, config):
        return architecture.Network(config)

    def add_metrics(self, metrics):
        """assign and add summary to a metric tensor"""
        for i,metric in enumerate(self.config.metrics):
            tf.summary.scalar(metric, metrics[i])

    def add_summaries(self,sess):
        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer_train = tf.train.SummaryWriter(self.config.logs_dir+"train", sess.graph)
        self.summary_writer_val = tf.train.SummaryWriter(self.config.logs_dir+"val", sess.graph)
    
    def write_summary(self,sess,summary_writer, metric_values, step, feed_dict):
        summary = self.summary.merged_summary
        #feed_dict[self.loss]=loss
        feed_dict[self.metrics]=metric_values
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()


    def run_epoch(self, sess, data, train_op=None, summary_writer=None,verbose=1000):
        if not train_op :
            train_op = tf.no_op()
            keep_prob = 1
        else:
            keep_prob = self.config.architecture._dropout
            
        # And then after everything is built, start the training loop.
        total_loss, label_loss = [], []
        grads, f1_micro, f1_macro = [], [], []
        total_steps = sum(1 for x in dataset.next_batch(self.config.batch_size,self.config.num_steps))	
	#Sets to state to zero for a new epoch
        state = self.arch.initial_state.eval()
        for step, (input_batch, label_batch) in enumerate(
            dataset.next_batch(self.config.batch_size,self.config.num_steps)):

            #print("\n\n\nActualLabelCount: ", input_batch, label_batch, label_batch_2, seq_len, np.sum(label_batch_2, axis=2))
            feed_dict = self.create_feed_dict(self.filter_inputs(input_batch, label_batch))
            feed_dict[self.keep_prob] = keep_prob
	    #feed_dict[self.arch.initial_state] = state 
	    
	    #Writes loss summary @last step of the epoch
            if (step+1) < total_steps:
                _, loss_value, state, pred_labels = sess.run([train_op, self.loss, self.arch.final_state, self.arch.label_sigmoid], feed_dict=feed_dict)
            else:
                _, loss_value, state, summary, pred_labels = sess.run([train_op, self.loss, self.arch.final_state,self.summary,self.arch.label_sigmoid], feed_dict=feed_dict)
                if summary_writer != None:
                    summary_writer.add_summary(summary,self.arch.global_step.eval(session=sess))
                    summary_writer.flush()
            #print(loss_value)
            total_loss.append(loss_value[0])
            label_loss.append(loss_value[2])
            #print(loss_value[5])
            grads.append(np.mean(loss_value[5][0]))
           

            #print("\n\n\nPredLabels:", pred_labels)
            if verbose and step % verbose == 0:
                if self.config.solver._curr_label_loss:
                    # metrics = perf.evaluate(pred_labels, label_batch_2, 0)
                    metrics = self.predict_results(sess, dataset.labels)
                    self.add_metrics(metrics)
                    f1_micro.append(metrics[3])
                    f1_macro.append(metrics[4])
                print('%d/%d : label = %0.4f : micro-F1 = %0.3f : macro-F1 = %0.3f : grads = %0.12f'%(step, total_steps,  np.mean(label_loss), np.mean(f1_micro), np.mean(f1_macro), np.mean(grads)), end="\r")
                sys.stdout.flush()
            
        if verbose:
            sys.stdout.write('\r')
        return np.mean(total_loss), np.mean(f1_micro), np.mean(f1_macro)

    def fit_outer(self, sess):
        while True: #put condition
            sess.run(self.init) #reset all weights
            print("------ Graph Reset | Next iteration -----")
            best_step, losses = fit(self, sess) #Train with current distribution of labels

            checkpoint_file = self.config.ckpt_dir + 'checkpoint'
            self.saver.restore(sess, checkpoint_file) #restore the best parameters        
            new_labels = self.predict_results(sess, data='test') #Get new predictions for unlabeled nodes

            self.update_labels(new_labels) #Update the labels with 
            

    def fit(self, sess):
        #define parametrs for early stopping early stopping
        max_epochs           = self.config.max_epochs
        patience             = self.config.patience              # look as this many examples regardless
        patience_increase    = self.config.patience_increase     # wait this much longer when a new best is found
        improvement_threshold= self.config.improvement_threshold # a relative improvement of this much is
                                                                 # considered significant
        
        # go through this many minibatches before checking the network on the validation set
        # Here we check every epoch
        validation_loss = 1e6
        done_looping = False
        step         = 1
        best_step    = -1
        losses       = []
        learning_rate= self.config.solver._parameters['learning_rate']
        #sess.run(self.init) #DO NOT DO THIS!! Doesn't restart from checkpoint
        while (step <= self.config.max_epochs) and (not done_looping):
            sess.run([self.step_incr_op])
            epoch = self.arch.global_step.eval(session=sess)

            start_time = time.time()
            average_loss, tr_micro, tr_macro = self.run_epoch(sess, data='train',train_op=self.train,summary_writer=self.summary_writer_train)
            duration = time.time() - start_time

            if (epoch % self.config.val_epochs_freq == 0):
                val_micro, val_macro = self.predict_results(data='val')

                print('\nEpoch %d: tr_loss = %.2f, val_loss = %.2f || tr_micro = %.2f, val_micro = %.2f || tr_macro = %.2f, val_macro = %.2f  (%.3f sec)'
                      % (epoch, average_loss, val_loss, tr_micro, val_micro, tr_macro, val_macro, duration))
                	
                # Save model only if the improvement is significant
                if (val_micro < validation_loss * improvement_threshold) and (epoch > self.config.save_epochs_after):
                    patience = max(patience, epoch * patience_increase)
                    validation_loss = val_micro
                    checkpoint_file = self.config.ckpt_dir + 'checkpoint'
                    self.saver.save(sess, checkpoint_file, global_step=epoch)
                    best_step = epoch
                    patience = epoch + max(self.config.val_epochs_freq,self.config.patience_increase)
                    print('best step %d'%(best_step))
		
                elif val_loss > validation_loss * improvement_threshold:
                    patience = epoch - 1

            else:
                print('Epoch %d: loss = %.2f pp = %.2f (%.3f sec)' % (epoch, average_loss, tr_pp, duration))

            if (patience <= epoch):
		#config.val_epochs_freq = 2
                learning_rate = learning_rate / 10
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                patience = epoch + max(self.config.val_epochs_freq,self.config.patience_increase)
                print('--------- Learning rate dropped to: %f'%(learning_rate))		
                if learning_rate <= 0.0000001:
                    print('Stopping by patience method')
                    done_looping = True

            losses.append(average_loss) 
            step += 1

        return losses, best_step



########END OF CLASS MODEL#####################################

def init_Model(config):
    tf.reset_default_graph()
    with tf.variable_scope('RNNLM',reuse=None) as scope:
        model = RNNLM_v1(config)
    
    tfconfig = tf.ConfigProto( allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    if config.retrain:
        load_ckpt_dir = config.ckpt_dir
        print('--------- Loading variables from checkpoint if available')
    else:
        load_ckpt_dir = ''
        print('--------- Training from scratch')
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir,config=tfconfig)
    return model, sess

def train_DNNModel():
    #global cfg
    print('############## Training Module ')
    config = deepcopy(cfg)
    model,sess = init_Model(config)
    with sess:
	    model.add_summaries(sess)
	    losses, best_step = model.fit_outer(sess)
    return losses


def execute():
    with tf.device('/gpu:0'):
        err = train_DNNModel() 
        return err
    
if __name__ == "__main__":
    #remove parameter dictionary
    
    meta_param = {#('dataset_name',):['blogcatalog_ncc'],
                  #('solver', 'learning_rate'): [0.001],
                  #('retrain',): [False],
                  ('debug',): [False],
                  ('max_epochs',): [1000]
    }

    variations = len(meta_param[('debug',)])

    #Make sure number of variants are equal
    for k,v in meta_param.items():
        assert len(v) == variations           
    
    
    for idx in range(variations):        
        for k,vals in meta_param.items():
            x = cfg
            if len(k) > 1:
                x = getattr(x, k[0])
            setattr(x, k[-1], vals[idx])
            print(k[-1], vals[idx])

        cfg.create(cfg.dataset_name)#"run-"+str(idx))
        cfg.init2()

        #All set... GO! 
        execute()
        print('\n\n ===================== \n\n')
