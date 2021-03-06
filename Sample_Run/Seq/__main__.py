from __future__ import print_function

import sys
import time
from copy import deepcopy

import numpy as np
import tensorflow as tf

import Config as conf
import Eval_Calculate_Performance as perf
from Eval_utils import write_results
import network as architecture
from blogDWdata import DataSet

cfg = conf.Config()


class RNNLM_v1(object):
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.arch = self.add_network(config)

        self.rnn_outputs = self.arch.predict(self.data_placeholder, self.data_placeholder2,
                                             self.keep_prob_in, self.keep_prob_out,self.label_in)
        self.outputs     = self.arch.projection(self.rnn_outputs)
        self.loss        = self.arch.loss(self.outputs, self.label_placeholder)
        self.optimizer   = self.config.solver._optimizer
        self.train       = self.arch.training(self.loss, self.optimizer)

        self.saver        = tf.train.Saver()
        self.summary      = tf.summary.merge_all()
        self.step_incr_op = self.arch.global_step.assign(self.arch.global_step + 1)
        self.init         = tf.global_variables_initializer()

    def bootstrap(self, sess, data, label_in):
        for step, (input_batch, input_batch2, seq, label_batch, tot) in enumerate(
                self.dataset.next_batch(data, batch_size=512, shuffle=False)):
            feed_dict = self.create_feed_dict(input_batch, input_batch2, label_batch, label_in)
            feed_dict[self.keep_prob_in] = 1
            feed_dict[self.keep_prob_out] = 1
            # feed_dict[self.arch.initial_state] = state
            pred_labels = sess.run([self.arch.label_sigmoid], feed_dict=feed_dict)
            self.dataset.accumulate_label_cache(pred_labels, seq)

            #print('%d/%d'%(step,tot), end="\r")
            #sys.stdout.flush()

        self.dataset.update_label_cache()

    def predict_results(self, sess, data, return_labels=False):
        labels_orig, labels_pred = [], []
        for node in np.where(self.dataset.get_nodes(data))[0]:
            # print('====',self.dataset.label_cache[node])
            labels_orig.append(self.dataset.all_labels[node])
            labels_pred.append(self.dataset.label_cache[node])

        if return_labels:
            return labels_pred
        else:
            return perf.evaluate(labels_pred, labels_orig, 0)

    def load_data(self):
        # Get the 'encoded data'
        self.dataset = DataSet(self.config)
        debug = self.config.debug
        if debug:
            print('##############--------- Debug mode [NOT IMPLEMENTED] ')
            num_debug = (self.config.num_steps + 1) * 128

        self.config.data_sets._len_vocab = self.dataset.all_features.shape[0]
        self.config.data_sets._len_labels = self.dataset.all_labels.shape[1]
        self.config.data_sets._len_features = self.dataset.all_features.shape[1]
        self.config.num_steps = self.dataset.all_walks.shape[1]

        print('--------- Project Path: ' + self.config.codebase_root_path + self.config.project_name)
        print('--------- Total number of nodes: ' + str(self.config.data_sets._len_vocab))
        print('--------- Walk length: ' + str(self.config.num_steps))
        print('--------- Label Length: ' + str(self.config.data_sets._len_labels))
        print('--------- Feature Length: ' + str(self.config.data_sets._len_features))
        print('--------- Train nodes: ' + str(np.sum(self.dataset.train_nodes)))
        print('--------- Val nodes: ' + str(np.sum(self.dataset.val_nodes)))
        print('--------- Test nodes: ' + str(np.sum(self.dataset.test_nodes)))

    def add_placeholders(self):
        self.data_placeholder = tf.placeholder(tf.float32,
                                               shape=[self.config.num_steps, None, self.config.data_sets._len_features],
                                               name='Input')
        self.data_placeholder2 = tf.placeholder(tf.float32,
                                                shape=[self.config.num_steps, None, self.config.data_sets._len_labels],
                                                name='label_inputs')
        self.label_placeholder = tf.placeholder(tf.float32, shape=[None, self.config.data_sets._len_labels],
                                                name='Target')
        self.keep_prob_in = tf.placeholder(tf.float32, name='keep_prob_in')
        self.keep_prob_out = tf.placeholder(tf.float32, name='keep_prob_out')
        self.label_in = tf.placeholder(tf.bool, name='label_input_condition')

    def create_feed_dict(self, input_batch, input_batch2, label_batch, label_in):
        feed_dict = {
            self.data_placeholder: input_batch,
            self.data_placeholder2: input_batch2,
            self.label_placeholder: label_batch,
            self.label_in: label_in
        }
        return feed_dict

    def add_network(self, config):
        return architecture.Network(config)

    def add_metrics(self, metrics):
        """assign and add summary to a metric tensor"""
        for i, metric in enumerate(self.config.metrics):
            tf.summary.scalar(metric, metrics[i])

    def print_metrics(self, inp):
        for idx, item in enumerate(inp):
            print(self.config.metrics[idx], ": ", item)

    def add_summaries(self, sess):
        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer_train = tf.train.SummaryWriter(self.config.logs_dir + "train", sess.graph)

    def write_summary(self, sess, summary_writer, metric_values, step, feed_dict):
        summary = self.summary.merged_summary
        # feed_dict[self.loss]=loss
        feed_dict[self.metrics] = metric_values
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()

    def run_epoch(self, sess, data, label_in, train_op=None, summary_writer=None, verbose=50):
        #Optimize the objective for one entire epoch via mini-batches
        
        if not train_op:
            train_op = tf.no_op()
            keep_prob_in = 1
            keep_prob_out = 1
        else:
            keep_prob_in = self.config.mRNN._keep_prob_in
            keep_prob_out = self.config.mRNN._keep_prob_out

        total_loss, label_loss = [], []
        gr_H, gr_I, gr_LI, f1_micro, f1_macro, accuracy = [],[], [], [], [], []
        # Sets to state to zero for a new epoch
        # state = self.arch.initial_state.eval()
        for step, (input_batch, input_batch2, seq, label_batch, tot) in enumerate(
                self.dataset.next_batch(data, self.config.batch_size, shuffle=True)):

            # print("\n\n\nActualLabelCount: ", np.shape(input_batch), np.shape(input_batch2), np.shape(label_batch), np.shape(seq))
            feed_dict = self.create_feed_dict(input_batch, input_batch2, label_batch, label_in)
            feed_dict[self.keep_prob_in] = keep_prob_in
            feed_dict[self.keep_prob_out] = keep_prob_out
            # feed_dict[self.arch.initial_state] = state

            # Writes loss summary @last step of the epoch
            if (step + 1) < tot:
                _, loss_value, pred_labels = sess.run([train_op, self.loss, self.arch.label_sigmoid],
                                                      feed_dict=feed_dict)
            else:
                _, loss_value, summary, pred_labels = sess.run(
                    [train_op, self.loss, self.summary, self.arch.label_sigmoid], feed_dict=feed_dict)
                if summary_writer != None:
                    summary_writer.add_summary(summary, self.arch.global_step.eval(session=sess))
                    summary_writer.flush()
            # print(loss_value)
            total_loss.append(loss_value[0])
            label_loss.append(loss_value[1])
            gr_H.append(np.mean(loss_value[2][0]))
            gr_I.append(np.mean(loss_value[2][1]))
            gr_LI.append(np.mean(loss_value[2][2]))

            # print("\n\n\nPredLabels:", pred_labels)
            if verbose and step % verbose == 0:
                metrics = [0] * 10
                if self.config.solver._curr_label_loss:
                    metrics = perf.evaluate(pred_labels, label_batch, 0)
                    f1_micro.append(metrics[3])
                    f1_macro.append(metrics[4])
                    accuracy.append(metrics[-1])
                print('%d/%d : label = %0.4f : micro-F1 = %0.3f : accuracy = %0.3f : gr_H = %0.12f : gr_I = %0.12f : gr_LI = %0.12f'
                      % (step, tot, np.mean(label_loss), np.mean(f1_micro),
                         np.mean(accuracy), np.mean(gr_H), np.mean(gr_I), np.mean(gr_LI)), end="\r")
                sys.stdout.flush()

        if verbose:
            sys.stdout.write('\r')
        return np.mean(total_loss), np.mean(f1_micro), np.mean(f1_macro), np.mean(accuracy)


    def fit(self, sess, label_in):
        #Controls how many time to optimize the function before making next label prediction
        average_loss, tr_micro, tr_macro, tr_accuracy = 0,0,0,0
        for step in range( self.config.max_inner_epochs):
            average_loss, tr_micro, tr_macro, tr_accuracy = self.run_epoch(sess, data='train', label_in=label_in,
                                                              train_op=self.train,
                                                              summary_writer=self.summary_writer_train)
        #return last evaluated loasses
        return average_loss, tr_micro, tr_macro, tr_accuracy 

    def fit_outer(self, sess):
        #[!!!IMP!!!] ideally, should also save the predicted labels to reload the performance later
        
        # define parametrs for early stopping early stopping
        max_epochs = self.config.max_outer_epochs
        patience = self.config.patience  # look as this many examples regardless
        patience_increase = self.config.patience_increase  # wait this much longer when a new best is found
        improvement_threshold = self.config.improvement_threshold  # a relative improvement of this much is considered significant

        validation_loss = 1e6
        done_looping = False
        step = 1
        best_step = -1
        losses = []
        learning_rate = self.config.solver.learning_rate
        label_in = None  # Ignore the label inputs during bootstrap | first run
        # sess.run(self.init) #DO NOT DO THIS!! Doesn't restart from checkpoint
        while (step <= self.config.max_outer_epochs) and (not done_looping):

                        
            #sess.run([self.step_incr_op])
            epoch = step#self.arch.global_step.eval(session=sess)


            print("------ Graph Reset | Next iteration -----")            
            #sess.run(self.init)  # reset all weights
            print([v for v in tf.trainable_variables()]) #Just to monitor the trainable variables in tf graph
            start_time = time.time()
            #Fit the model to predict best possible labels given the current estimates of unlabeled values
            average_loss, tr_micro, tr_macro, tr_accuracy = self.fit(sess, label_in)
            duration = time.time() - start_time

            #Make this true after first round of trainig has been done
            label_in = True

            if (epoch % self.config.val_epochs_freq == 0):
                # Get new estimates of unlabeled validation nodes
                #the actual inputs that resulted in this new result
                old_labels = deepcopy(self.dataset.label_cache)
                self.bootstrap(sess, data='all', label_in=label_in)  
                metrics = self.predict_results(sess, data='val') #evaluate performance for validation set
                val_micro, val_macro, val_loss, val_accuracy = metrics[3], metrics[4], metrics[-2], metrics[-1]

                print(
                    '\nEpoch %d: tr_loss = %.2f, val_loss %.2f || tr_micro = %.2f, val_micro = %.2f || tr_acc = %.2f, val_acc = %.2f  (%.3f sec)'
                    % (epoch, average_loss, val_loss, tr_micro, val_micro, tr_accuracy, val_accuracy, duration))

                # Save model only if the improvement is significant
                if (val_loss< validation_loss * improvement_threshold) and (epoch > self.config.save_epochs_after):
                    #patience = max(patience, epoch * patience_increase)
                    validation_loss = val_loss
                    
                    #checkpoint_file = self.config.ckpt_dir + 'checkpoint'
                    #self.saver.save(sess, checkpoint_file, global_step=epoch)
                    self.saver.save(sess, self.config.ckpt_dir + 'last-best')
                    np.save( self.config.ckpt_dir + 'last-best_labels.npy',old_labels)
                    
                    best_step = epoch
                    patience = epoch + max(self.config.val_epochs_freq, self.config.patience_increase)
                    print('best step %d' % (best_step))

                #elif val_loss > validation_loss * improvement_threshold:
                #    patience = epoch - 1

                #Get predictions for test nodes
                #These metrics are not used anywhere :)
                self.print_metrics( self.predict_results(sess, data='test'))

            else:
                print('Epoch %d: loss = %.2f (%.3f sec)' % (epoch, average_loss, duration))

            """
            #Uncomment this if weights are NOT re-initialized after bootstrap for new labels
            #If weights are re-initialised then we can't reduce the learning rate immendiately
            """

            if patience <= epoch:
                # config.val_epochs_freq = 2
                learning_rate = learning_rate / 10
                self.optimizer = self.config.solver.opt(learning_rate)
                patience = epoch + max(self.config.val_epochs_freq, self.config.patience_increase)
                print('--------- Learning rate dropped to: %f' % (learning_rate))
 
                self.saver.restore(sess, tf.train.latest_checkpoint(self.config.ckpt_dir))
        	self.dataset.label_cache = np.load(self.config.ckpt_dir + 'last-best_labels.npy').item()
 
                if learning_rate <= 0.0000001:
                    print('Stopping by patience method')
                    done_looping = True


            losses.append(average_loss)
            step += 1

        #End of Training 


        #Create new session and load the last best
        #tfconfig = tf.ConfigProto(allow_soft_placement=True)
        #tfconfig.gpu_options.allow_growth = True
        #sess = tf.Session(config=tfconfig)
        #new_saver = tf.train.import_meta_graph(self.config.ckpt_dir + 'last-best.meta')
        #new_saver.restore(sess, tf.train.latest_checkpoint(self.config.ckpt_dir))
        # checkpoint_file = self.config.ckpt_dir + 'checkpoint'
        self.saver.restore(sess, tf.train.latest_checkpoint(self.config.ckpt_dir)) #restore the best parameters
        self.dataset.label_cache = np.load(self.config.ckpt_dir + 'last-best_labels.npy').item()     
   
        self.bootstrap(sess, data='all', label_in=label_in)  # Get new estimates of unlabeled nodes
        metrics  = self.predict_results(sess, data='test')
        self.print_metrics(metrics)  # Get predictions for test nodes

        return metrics


########END OF CLASS MODEL#####################################

def init_Model(config):
    tf.reset_default_graph()
    with tf.variable_scope('RNNLM', reuse=None) as scope:
        model = RNNLM_v1(config)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    #if config.retrain:
    #    load_ckpt_dir = config.ckpt_dir
    #    print('--------- Loading variables from checkpoint if available')
    #else:
    load_ckpt_dir = ''
    print('--------- Training from scratch')
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir, config=tfconfig)
    return model, sess


def train_DNNModel():
    # global cfg
    print('############## Training Module ')
    config = deepcopy(cfg)
    model, sess = init_Model(config)
    with sess:
        model.add_summaries(sess)
        metrics = model.fit_outer(sess)
        return metrics

def execute():
    with tf.device('/gpu:1'):
        metrics = train_DNNModel()
        return metrics


if __name__ == "__main__":
    meta_param = {  # ('dataset_name',):['blogcatalog_ncc'],
        # ('solver', 'learning_rate'): [0.001],
        ('train_fold',): [1,2,3,4,5]
    }

    variations = len(meta_param.values()[0])
    # Make sure number of variants are equal
    for k, v in meta_param.items():
        assert len(v) == variations

    all_results = {cfg.train_percent: {}}
    for idx in range(variations):
        for k, vals in meta_param.items():
            x = cfg
            if len(k) > 1:
                x = getattr(x, k[0])
            setattr(x, k[-1], vals[idx])
            print(k[-1], vals[idx])

        cfg.create(cfg.dataset_name)  # "run-"+str(idx))
        cfg.init2()
        print("=====Configurations=====\n", cfg.__dict__)

        # All set... GO!
        metrics = execute()
        all_results[cfg.train_percent][meta_param[('train_fold',)][idx]] = metrics
        print('\n\n ===================== \n\n')

        write_results(cfg, all_results)

