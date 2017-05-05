
# !/usr/bin/env python
from __future__ import print_function

import sys
import time
import os
from copy import deepcopy

import numpy as np
import tensorflow as tf

import Config as conf
import Eval_Calculate_Performance as perf
from Eval_utils import write_results, plotit
import network as architecture
import argparse
from blogDWdata import DataSet
import threading


os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"]="0"

np.random.seed(1234)
tf.set_random_seed(1234)

class RNNLM_v1(object):
    def __init__(self, config):
        self.config = config
        self.load_data()
        self.add_placeholders()
        self.arch = self.add_network(config)
        self.change = 0
        self.fluctuations = {}

        self.Q = tf.FIFOQueue(capacity=50, dtypes=[tf.float32, tf.float32, tf.float32, tf.int32, tf.int32])
        self.enqueue_op = self.Q.enqueue_many([self.x_attr_placeholder, self.x_labels_placeholder, self.y_label_placeholder,
                                               self.x_lengths_placeholder, self.node_id_placeholder])
        self.x_attr, self.x_labels, self.y_labels, self.x_lengths, self.node_id  = self.Q.dequeue()

        self.rnn_outputs, context, x_attr_reduced = self.arch.predict(self.x_attr, self.x_labels, self.keep_prob_in, self.keep_prob_out, self.x_lengths)
        self.attn_outputs= self.arch.attention(self.rnn_outputs, context = context)
        self.outputs     = self.arch.projection(self.attn_outputs, x_attr_reduced, self.x_labels)
        self.loss        = self.arch.loss(self.outputs, self.y_labels, self.wce_placeholder)
        self.optimizer   = self.config.solver._optimizer
        train            = self.arch.custom_training(self.loss, self.optimizer, self.config.batch_size) #self.arch.training(self.loss, self.optimizer)
        self.reset_grads, self.accumulate_op, self.update_op  = train

        self.saver        = tf.train.Saver()
        self.summary      = tf.summary.merge_all()
        self.step_incr_op = self.arch.global_step.assign(self.arch.global_step + 1)
        self.init         = tf.global_variables_initializer()

    def load_and_enqueue(self, sess, data):
        for idx, (input_batch, input_batch2, label, x_lengths, node_id) in enumerate(self.dataset.walks_generator(data)):
            feed_dict = self.create_feed_dict([input_batch], [input_batch2], [label], [x_lengths], [node_id])
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def bootstrap(self, sess, data, update=True):
        alpha = self.config.solver.label_update_rate
        if len(self.dataset.label_cache.items()) <= 1: alpha =1.0 #First update
        depth_sum, attn_sum = 0, 0

        update_cache = {}

        start = time.time()
        load_time, run_time, update_time = 0, 0, 0

        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tot = np.sum(self.dataset.get_nodes(data))

        for step in range(tot):
            feed_dict = {self.keep_prob_in: 1, self.keep_prob_out: 1}
            node_id, self.attn_values, pred_labels = sess.run([self.node_id, self.arch.attn_vals, self.arch.label_preds], feed_dict=feed_dict)
            t2 = time.time()
            run_time += t2 - start

            new = pred_labels[0] #np.mean(pred_labels[start:start+count], axis=0)
            old = np.array(self.dataset.label_cache.get(node_id, self.dataset.all_labels[0]))
            updated = (1-alpha)*old + alpha*new
            self.change += np.mean((updated - old) ** 2)
            update_cache[node_id] = updated #store all the updates in temporary dict

            start = time.time()
            update_time += start - t2

        coord.request_stop()
        coord.join(threads)

        #print("Depth sums: ", depth_sum)
        #print("Attention: ", self.attn_values)

        #print("Change in label: :", np.sqrt(self.change/self.config.data_sets._n_nodes)*100)
        self.change = 0

        #print("Step: %d :: Load time: %f :: Run time: %f :: Update time: %f"%(step, load_time, run_time, update_time))

        #Assign the predicted labels to label_cache
        if update:
            print("========== Label updated ============= \n")
            self.dataset.label_cache = update_cache

        return update_cache

    def predict_results(self, sess, data, return_labels=False, preds=None):
        if preds == None:
            preds = self.dataset.label_cache

        labels_orig, labels_pred = [], []
        for node in np.where(self.dataset.get_nodes(data))[0]:
            # print('====',self.dataset.label_cache[node])
            labels_orig.append(self.dataset.all_labels[node])
            labels_pred.append(preds[node])

        if return_labels:
            return perf.evaluate(labels_pred, labels_orig, 0), labels_pred
        else:
            return perf.evaluate(labels_pred, labels_orig, 0)


    def load_data(self):
        # Get the 'encoded data'
        self.dataset = DataSet(self.config)
        self.config.data_sets._len_labels = self.dataset.n_labels
        self.config.data_sets._len_features = self.dataset.n_features
        self.config.data_sets._multi_label = self.dataset.multi_label
        self.config.data_sets._n_nodes = self.dataset.n_nodes
        self.config.num_steps = self.dataset.diameter + 1
        print('--------- Project Path: ' + self.config.codebase_root_path + self.config.project_name)

    def add_placeholders(self):
        #0th axis should have same size for all tensord in the Queue
        self.x_attr_placeholder = tf.placeholder(tf.float32, name='Input', shape=[1, self.config.num_steps, None, self.config.data_sets._len_features])
        self.x_labels_placeholder = tf.placeholder(tf.float32, name='label_inputs', shape=[1, self.config.num_steps, None, self.config.data_sets._len_labels])
        self.x_lengths_placeholder = tf.placeholder(tf.int32, name='walk_lengths',shape=[1,None])

        self.y_label_placeholder = tf.placeholder(tf.float32, name='Target', shape=[1, 1, self.config.data_sets._len_labels])
        self.node_id_placeholder = tf.placeholder(tf.int32, name='node_id', shape=[1])

        self.keep_prob_in = tf.placeholder(tf.float32, name='keep_prob_in')
        self.keep_prob_out = tf.placeholder(tf.float32, name='keep_prob_out')
        self.wce_placeholder = tf.placeholder(tf.float32, shape=[self.config.data_sets._len_labels], name='Cross_entropy_weights')

    def create_feed_dict(self, input_batch, input_batch2, label_batch, x_lengths, node_id):
        feed_dict = {
            self.x_attr_placeholder: input_batch,
            self.x_labels_placeholder: input_batch2,
            self.y_label_placeholder: label_batch,
            self.x_lengths_placeholder: x_lengths,
            self.node_id_placeholder: node_id
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

    def run_epoch2(self, sess, data, train_op=None, summary_writer=None, verbose=50):
        # Test input data

        keep_prob_in = self.config.mRNN._keep_prob_in
        keep_prob_out = self.config.mRNN._keep_prob_out

        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        for i in range(3):
            feed_dict = {self.keep_prob_in: keep_prob_in, self.keep_prob_out: keep_prob_out,
                         self.wce_placeholder: self.dataset.wce}
            a,b,c,d,e = sess.run([self.x_attr, self.x_labels, self.y_labels, self.x_lengths, self.node_id ], feed_dict=feed_dict)
            #print("Data: ", a,b,c,d,e)


        coord.request_stop()
        coord.join(threads)

    def run_epoch(self, sess, data, train_op, summary_writer=None, verbose=50):
        # Optimize the objective for one entire epoch via mini-batches

        total_loss, gradients, f1_micro, f1_macro, accuracy, bae = [], [], [], [], [], []
        # Sets to state to zero for a new epoch
        # state = self.arch.initial_state.eval()

        sess.run([self.reset_grads])  # Reset grad accumulator at the beginning
        pred_label_accum, target_label_accum, step = [], [], 0

        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)
        tot = np.sum(self.dataset.get_nodes(data))

        while step < tot:
            step += 1
            feed_dict = {self.keep_prob_in: self.config.mRNN._keep_prob_in,
                         self.keep_prob_out: self.config.mRNN._keep_prob_out,
                         self.wce_placeholder: self.dataset.wce}

            # Writes loss summary @last step of the epoch
            if step < tot:
                grads, loss_value, pred_labels, target_label = sess.run(
                    [train_op, self.loss, self.arch.label_preds, self.y_labels],
                    feed_dict=feed_dict)
            else:
                grads, loss_value, summary, pred_labels, target_label = sess.run(
                    [train_op, self.loss, self.summary, self.arch.label_preds, self.y_labels], feed_dict=feed_dict)
                if summary_writer != None:
                    summary_writer.add_summary(summary, self.arch.global_step.eval(session=sess))
                    summary_writer.flush()

            gradients.append([np.max(np.abs(item)) for item in grads])  # get the absolute maximum gradient to each variable
            total_loss.append(loss_value)

            pred_label_accum.append(pred_labels)
            target_label_accum.append(target_label)

            if verbose and step % verbose == 0:
                metrics = perf.evaluate(pred_label_accum, target_label_accum, 0)
                pred_label_accum, target_label_accum = [], []
                f1_micro.append(metrics[3])
                f1_macro.append(metrics[4])
                accuracy.append(metrics[-1])
                bae.append(metrics[-3])
                #print('%d/%d : loss = %0.4f : micro-F1 = %0.3f : accuracy = %0.3f : bae = %0.3f'
                #       % (step, tot, np.mean(loss_value), np.mean(f1_micro), np.mean(accuracy), np.mean(bae)))#, end="\r")


            if step % self.config.batch_size == 0 or step == tot:
                # Print Gradients for each trainable weight
                if self.config.solver.gradients:
                    print("%d/%d :: " % (step, tot), end="")
                    for var, val in zip(['-'.join(k.name.split('/')[-2:]) for k in tf.trainable_variables()],
                                        np.mean(gradients, axis=0)):
                        print("%s :: %.8f  " % (var, val/self.config.batch_size), end="")
                    print("\n")
                sys.stdout.flush()

                # Update gradients after batch_size or at the end of the current epoch\
                #print("Gradients updated at step: %d\n"%(step))
                sess.run([self.update_op])
                sess.run([self.reset_grads])

        coord.request_stop()
        coord.join(threads)

        return np.mean(total_loss), np.mean(f1_micro), np.mean(f1_macro), np.mean(accuracy), np.mean(bae)

    def fit(self, sess, epoch, best_val_loss):
        # Controls how many time to optimize the function before making next label prediction
        patience_increase = self.config.patience_increase  # wait this much longer when a new best is found
        improvement_threshold = self.config.improvement_threshold  # a relative improvement of this much is considered significant
        patience = epoch + max(self.config.val_epochs_freq, self.config.patience_increase)

        for i in range(self.config.max_outer_epochs): #change this
            start_time = time.time()
            average_loss, tr_micro, tr_macro, tr_accuracy, tr_bae = self.run_epoch(sess, data='train', train_op=self.accumulate_op,
                                                                           summary_writer=self.summary_writer_train)
            duration = time.time() - start_time

            #print("Tr_micro: %f :: Tr_macro: %f :: Tr_accuracy: %f :: Tr_bae: %f :: Time: %f"%(tr_micro, tr_macro, tr_accuracy, tr_bae, duration))
            if (epoch % self.config.val_epochs_freq == 0):

                s = time.time()
                pred_labels = self.bootstrap(sess, data='val', update=False)
                #print('Bootstrap time: ', time.time() - s)

                #pred_labels = self.dataset.get_update_cache()
                metrics = self.predict_results(sess, data='val', preds=pred_labels)  # evaluate performance for validation set
                val_micro, val_macro, val_loss, val_accuracy = metrics[3], metrics[4], metrics[-2], metrics[-1]
                val_loss = 1-val_accuracy #using accuracy as stopping criteria instead of cross-loss

                print('Epoch %d: tr_loss = %.2f, val_loss %.2f || tr_micro = %.2f, val_micro = %.2f || tr_acc = %.2f, val_acc = %.2f  (%.3f sec)'
                        %(epoch, average_loss, val_loss, tr_micro, val_micro, tr_accuracy, val_accuracy, duration))

                # Save model only if the improvement is significant
                if (val_loss < best_val_loss * improvement_threshold):
                    best_val_loss = val_loss
                    self.saver.save(sess, self.config.ckpt_dir + 'last-best')

                    patience = epoch + max(self.config.val_epochs_freq, patience_increase)
                    print('best step %d\n' % (epoch))

                if patience <= epoch:
                    break

            epoch +=1

        return epoch, best_val_loss

    def fit_outer(self, sess):
        # define parametrs for early stopping early stopping
        max_epochs = self.config.max_outer_epochs
        patience = self.config.patience  # look as this many examples regardless
        done_looping = False
        epoch = 1
        flag = self.config.boot_reset
        outer_epoch = 1
        learning_rate = self.config.solver.learning_rate
        best_loss = 1e6

        while (outer_epoch <= max_epochs) and (not done_looping):
            # sess.run([self.step_incr_op])
            # self.arch.global_step.eval(session=sess)
            if outer_epoch == 2 and flag: #reset after first bootstrap
                print("------ Graph Reset | First bootstrap done -----\n\n\n")
                sess.run(self.init)  # reset all weights
                flag = False
                best_loss = 1e6 #reset beset loss; under assumption that model ALWAYS will do better with pseudo labels than without

            print([v.name for v in tf.trainable_variables()], "\n")  # Just to monitor the trainable variables in tf graph

            # Fit the model to predict best possible labels given the current estimates of unlabeled values
            epoch, new_loss = self.fit(sess, epoch, best_loss)
            self.saver.restore(sess, tf.train.latest_checkpoint(self.config.ckpt_dir))  # Restore the best parameters
            outer_epoch +=1

            if new_loss >= best_loss:
                learning_rate = learning_rate / 2
                self.optimizer = self.config.solver.opt(learning_rate)
                print('--------- Learning rate dropped to: %f' % (learning_rate))

                if learning_rate <= 0.000001:
                    print('Stopping by patience method')
                    done_looping = True

            else:
                self.bootstrap(sess, data='all', update=True)
                best_loss = new_loss

                # Get predictions for test nodes
                test_metrics = self.predict_results(sess, data='test')
                self.print_metrics(test_metrics)

                #Additional book-kepping to correlate val vs test accuracies
                val_metrics = self.predict_results(sess, data='val')
                self.fluctuations[epoch] = {'val': val_metrics, 'test': test_metrics}


        # End of Training
        #self.saver.restore(sess, tf.train.latest_checkpoint(self.config.ckpt_dir))  # restore the best parameters

        metrics = self.predict_results(sess, data='test')
        self.print_metrics(metrics)  # Get predictions for test nodes

        return metrics, self.attn_values, self.dataset.label_cache, self.fluctuations


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


def train_DNNModel(cfg):
    with tf.device('/gpu:1'):
        print('############## Training Module ')
        config = deepcopy(cfg)
        model, sess = init_Model(config)
        with sess:
            model.add_summaries(sess)
            metrics, attn_values, preds, fluctuations = model.fit_outer(sess)
            return metrics, attn_values, preds, fluctuations


def get_argumentparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='/home/priyesh/Desktop/Codes/Sample_Run/', help="Base path for the code")
    parser.add_argument("--project", default='path_attn', help="Project folder")
    parser.add_argument("--dataset", default='cora', help="Dataset to evluate")
    parser.add_argument("--labels", default='labels_random', help="Label type")
    parser.add_argument("--percent", default=20, help="Training percent")
    parser.add_argument("--folds", default='1_2_3_4_5', help="Training folds")
    parser.add_argument("--retrain", default=True, help="Retrain flag")
    parser.add_argument("--debug", default=False, help="Debug flag")
    parser.add_argument("--save_after", default=0, help="Save after epochs", type=int)
    parser.add_argument("--val_freq", default=1, help="Validation frequency", type=int)
    parser.add_argument("--bin_upd", default=0, help="Binary updates for labels", type=int)
    parser.add_argument("--gradients", default=0, help="Print gradients of trainable variables", type=int)
    parser.add_argument("--max_depth", default=5, help="Maximum path depth", type=int)
    parser.add_argument("--max_outer", default=2, help="Maximum outer epoch", type=int)
    parser.add_argument("--max_inner", default=1, help="Maximum inner epoch", type=int)
    parser.add_argument("--pat", default=10, help="Patience", type=int)
    parser.add_argument("--pat_inc", default=10, help="Patience Increase", type=int)
    parser.add_argument("--folder_suffix", default='', help="folder name suffix")

    parser.add_argument("--batch_size", default=256, help="Batch size", type=int)
    parser.add_argument("--boot_epochs", default=1, help="Epochs for first bootstrap", type=int)
    parser.add_argument("--boot_reset", default=1, help="Reset weights after first bootstrap", type=int)
    parser.add_argument("--concat", default=0, help="Concat attribute to hidden state", type=int)
    parser.add_argument("--wce", default=0, help="Wrighted cross entropy", type=int)
    parser.add_argument("--pat_improve", default=0.9999, help="Improvement threshold for patience", type=float)
    parser.add_argument("--lr", default=0.001, help="Learning rate", type=float)
    parser.add_argument("--lu", default=0.75, help="Label update rate", type=float)
    parser.add_argument("--l2", default=1e-3, help="L2 loss", type=float)
    parser.add_argument("--opt", default='adam', help="Optimizer type (adam, rmsprop, sgd)")
    parser.add_argument("--cell", default='LSTM', help="RNN cell (LSTM, myLSTM, GRU, RNN)")
    parser.add_argument("--reduce", default=0, help="Reduce Attribute dimensions to", type=int)
    parser.add_argument("--hidden", default=16, help="Hidden units", type=int)
    parser.add_argument("--attention", default=0, help="Attention module (0: no, 1: HwC, 2: tanh(wH + wC))",
                        type=int)
    parser.add_argument("--drop_in", default=0.5, help="Dropout for input", type=float)
    parser.add_argument("--drop_out", default=0.75, help="Dropout for pre-final layer", type=float)

    parser.add_argument("--ssl", default=0, help="Semi-supervised loss", type=int)
    return parser


def main():
    parser = get_argumentparser()
    args = parser.parse_args()
    print("=====Configurations=====\n", args)
    cfg = conf.Config(args)

    #Meta-script for running codes **SEQUENTIALLY**
    meta_param = {  #('dataset_name',):['blogcatalog_ncc'],
        # ('solver', 'learning_rate'): [0.001],
        ('train_fold',): np.fromstring(args.folds, sep='_', dtype=int)
    }

    variations = len(meta_param.values()[0])
    # Make sure number of variants are equal
    for k, v in meta_param.items():
        assert len(v) == variations

    attention = {}
    all_results = {cfg.train_percent: {}}
    for idx in range(variations):
        for k, vals in meta_param.items():
            x = cfg
            if len(k) > 1:
                x = getattr(x, k[0])
            setattr(x, k[-1], vals[idx])
            print(k[-1], vals[idx])

        cfg.create(cfg.dataset_name + str(cfg.train_percent) + args.folder_suffix)  # "run-"+str(idx))
        cfg.init2()
        #print("=====Configurations=====\n", cfg.__dict__)

        # All set... GO!
        metrics, attention[idx], preds, fluctuations = train_DNNModel(cfg)
        all_results[cfg.train_percent][cfg.train_fold] = metrics
        print('\n\n ===== Attention \n', attention[idx])
        print('\n\n ===================== \n\n')

        np.save(cfg.results_folder+'labels-'+str(cfg.train_percent)+'-'+str(cfg.train_fold), preds)
        np.save(cfg.results_folder+'metrics-fluctuations-'+str(cfg.train_percent)+'-'+str(cfg.train_fold), fluctuations)

        write_results(cfg, all_results)
        if cfg.mRNN.attention:
            plotit(attention, 1, 'Depth', 'Values', 'Attention at Depth',cfg)


if __name__ == "__main__":
    main()
