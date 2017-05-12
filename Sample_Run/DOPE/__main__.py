import utils
from config import Config
from parser import Parser
from dataset import DataSet
from network import Network
import eval_performance as perf

import sys
import time
import pickle
import threading
import numpy as np
import tensorflow as tf
from os import path
from copy import deepcopy
from tabulate import tabulate


class DeepDOPE(object):

    def __init__(self, config):
        self.config = config
        self.patience = self.config.patience
        self.learning_rate = self.config.solver.learning_rate

        self.dataset = self.load_data()
        self.ph_lr, self.ph_keep_prob_in, self.ph_keep_prob_out, self.ph_wce, self.ph_batch_size = self.get_placeholders()

        # Setup Data Queue
        self.ph_ids, self.ph_x_attr, self.ph_x_labels, self.ph_x_lengths, self.ph_y_label, \
            self.ph_node_id = self.get_queue_placeholders()
        self.Q, self.enqueue_op, self.dequeue_op = self.setup_data_queues()
        self.ids, self.x_attr, self.x_labels, self.x_lengths, self.y_labels, self.node_id = self.dequeue_op

        self.arch = self.add_network(config)
        # Learn a representation for information diffusion across each path
        self.neighbor_data, self.NOI_x, self.N_neighbors = self.arch.get_path_data(self.x_attr, self.x_labels,
                                                                 self.x_lengths, self.ph_keep_prob_in, self.ph_keep_prob_out)
        # Get Node of Interest's data
        self.NOI_data = self.arch.get_NOI_data(self.NOI_x, self.ph_keep_prob_in)
        self.path_ensemble_outputs = self.arch.attentive_ensemble(self.NOI_data, self.neighbor_data)

        with tf.variable_scope('Predictions') as scope:
            self.att_prediction = self.arch.predict(self.NOI_data, None, self.ph_keep_prob_out)
            scope.reuse_variables()
            # Get individual path predictions
            self.path_predictions = self.arch.predict(self.neighbor_data, None, self.ph_keep_prob_out)
            # combine diffusion over different paths attentively based on NOI
            self.path_prediction = self.arch.predict(self.path_ensemble_outputs, None, self.ph_keep_prob_out)
            # combine the ensmebled path data with the NOI data and predict labels
            self.combined_prediction = self.arch.predict(self.NOI_data, self.path_ensemble_outputs, self.ph_keep_prob_out)
        self.predictions = [self.att_prediction, self.path_prediction, self.combined_prediction]

        #Losses
        self.consensus_loss = self.arch.consensus_loss(self.path_predictions, self.path_prediction)
        self.node_loss = self.arch.loss(self.att_prediction, self.y_labels, self.ph_wce)
        self.path_loss = self.arch.loss(self.path_prediction, self.y_labels, self.ph_wce)
        self.combined_loss = self.arch.loss(self.combined_prediction, self.y_labels, self.ph_wce)
        self.total_loss = self.combined_loss + self.config.solver.path_loss*self.path_loss + \
                          self.config.solver.node_loss*self.node_loss + \
                          self.config.solver.consensus_loss*self.consensus_loss
        self.losses = [self.node_loss, self.path_loss, self.combined_loss, self.total_loss, self.consensus_loss]

        #Optimizer
        self.optimizer = self.config.solver._optimizer(self.ph_lr)
        train = self.arch.custom_training(self.total_loss, self.optimizer, self.config.batch_size)
        self.reset_grads, self.accumulate_op, self.update_op = train
        #self.train = self.arch.training(self.loss, self.optimizer)

        self.saver = tf.train.Saver()
        self.summary = tf.summary.merge_all()
        self.step_incr_op = self.arch.global_step.assign(self.arch.global_step + 1)
        self.init = tf.global_variables_initializer()

    def load_and_enqueue(self, sess, data):
        for idx, (ids, x_attr, x_labels, x_lengths, label, node_id) in enumerate(self.dataset.walks_generator(data)):
            feed_dict = self.create_feed_dict([ids], [x_attr], [x_labels], [x_lengths], [label], [node_id])
            sess.run(self.enqueue_op, feed_dict=feed_dict)

    def load_data(self):
        # Get the 'encoded data'
        dataset = DataSet(self.config)
        self.config.data_sets._len_labels = dataset.n_labels
        self.config.data_sets._len_features = dataset.n_features
        self.config.data_sets._multi_label = dataset.multi_label
        self.config.data_sets._n_nodes = dataset.n_nodes
        self.config.num_steps = dataset.diameter + 1
        print('--------- Project Path: ' + self.config.codebase_root_path + self.config.project_name)
        return dataset

    def get_queue_placeholders(self):
        # 0th axis should have same size for all tensord in the Queue
        ids_placeholder = tf.placeholder(tf.int32, name='Walk_ids', shape=[1, self.config.num_steps, None])
        x_attr_placeholder = tf.placeholder(tf.float32, name='Input',
                                            shape=[1, self.config.num_steps, None, self.config.data_sets._len_features])
        x_labels_placeholder = tf.placeholder(tf.float32, name='label_inputs',
                                              shape=[1, self.config.num_steps, None, self.config.data_sets._len_labels])
        x_lengths_placeholder = tf.placeholder(tf.int32, name='walk_lengths', shape=[1, None])

        y_label_placeholder = tf.placeholder(tf.float32, name='Target', shape=[1, 1, self.config.data_sets._len_labels])
        node_id_placeholder = tf.placeholder(tf.int32, name='node_id', shape=[1])
        return ids_placeholder, x_attr_placeholder, x_labels_placeholder, x_lengths_placeholder, y_label_placeholder, node_id_placeholder

    def get_placeholders(self):
        lr = tf.placeholder(tf.float32, name='learning_rate')
        keep_prob_in = tf.placeholder(tf.float32, name='keep_prob_in')
        keep_prob_out = tf.placeholder(tf.float32, name='keep_prob_out')
        batch_size = tf.placeholder(tf.float32, name='batch_size')
        wce_placeholder = tf.placeholder(tf.float32, shape=[self.config.data_sets._len_labels], name='Cross_entropy_weights')
        return lr, keep_prob_in, keep_prob_out, wce_placeholder, batch_size

    def setup_data_queues(self):
        Q = tf.FIFOQueue(capacity=50, dtypes=[tf.int32, tf.float32, tf.float32, tf.int32, tf.float32, tf.int32])
        enqueue_op = Q.enqueue_many([self.ph_ids, self.ph_x_attr, self.ph_x_labels, self.ph_x_lengths,
                                     self.ph_y_label, self.ph_node_id])
        dequeue_op = Q.dequeue()
        return Q, enqueue_op, dequeue_op

    def create_feed_dict(self, ids, x_attr, x_labels, x_lengths, label_batch, node_id):
        feed_dict = {
            self.ph_ids: ids,
            self.ph_x_attr: x_attr,
            self.ph_x_labels: x_labels,
            self.ph_x_lengths: x_lengths,
            self.ph_y_label: label_batch,
            self.ph_node_id: node_id,
            self.ph_batch_size: self.config.batch_size}
        return feed_dict

    def add_network(self, config):
        return Network(config)

    def add_summaries(self, sess):
        # Instantiate a SummaryWriter to output summaries and the Graph.
        summary_writer_train = tf.summary.FileWriter(self.config.logs_dir + "train", sess.graph)
        summary_writer_val = tf.summary.FileWriter(self.config.logs_dir + "validation", sess.graph)
        summary_writer_test = tf.summary.FileWriter(self.config.logs_dir + "test", sess.graph)
        summary_writers = {'train': summary_writer_train, 'val': summary_writer_val, 'test': summary_writer_test}
        return summary_writers

    def add_metrics(self, metrics):
        """assign and add summary to a metric tensor"""
        for i, metric in enumerate(self.config.metrics):
            tf.summary.scalar(metric, metrics[i])

    def print_metrics(self, inp):
        for idx, item in enumerate(inp):
            print(self.config.metrics[idx], ": ", item)

    def run_epoch(self, sess, data, train_op=None, summary_writer=None, verbose=10, learning_rate=0):
        train = train_op
        if train_op is None:
            train_op = tf.no_op()
            keep_prob_in = 1
            keep_prob_out = 1
        else:
            keep_prob_in = self.config.mRNN._keep_prob_in
            keep_prob_out = self.config.mRNN._keep_prob_out

        # Set up all variables
        total_steps = np.sum(self.dataset.get_nodes(data))  # Number of Nodes to run through
        verbose = min(verbose, total_steps) - 1
        node_ids, gradients, targets, attn_values = [], [], [], []
        losses, predictions, metrics= dict(), dict(), dict()

        metrics['node'], metrics['path'], metrics['combined'] = [], [], []
        predictions['node'], predictions['path'], predictions['combined'] = [], [], []
        losses['node'], losses['path'], losses['combined'], losses['consensus'], losses['total'] = [], [], [], [], []

        ########################################################################################################
        feed_dict = {self.ph_keep_prob_in: keep_prob_in, self.ph_keep_prob_out: keep_prob_out,
                     self.ph_wce: self.dataset.wce, self.ph_lr: learning_rate}

        # Reset grad accumulator at the beginning
        sess.run([self.reset_grads], feed_dict=feed_dict)

        #Start Running Queue
        t = threading.Thread(target=self.load_and_enqueue, args=[sess, data])
        t.daemon = True
        t.start()
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(coord=coord)

        step = 0
        while step < total_steps:
            step += 1
            feed_dict = {self.ph_keep_prob_in: keep_prob_in, self.ph_keep_prob_out: keep_prob_out,
                         self.ph_wce: self.dataset.wce, self.ph_lr: learning_rate}

            if step < total_steps - 1:
                id, grads, t_losses, t_pred_probs, target_label, t_attn_values = \
                    sess.run([self.node_id, train_op, self.losses, self.predictions, self.y_labels,
                              self.arch.attn_values], feed_dict=feed_dict)
            else:
                summary, id, grads, t_losses, t_pred_probs, target_label, t_attn_values = \
                    sess.run([self.summary, self.node_id, train_op, self.losses, self.predictions, self.y_labels,
                              self.arch.attn_values], feed_dict=feed_dict)
                if summary_writer is not None:
                    summary_writer.add_summary(summary, self.arch.global_step.eval(session=sess))
                    summary_writer.flush()

            node_ids.append(id)
            # Accumulate attention values
            attn_values.append(t_attn_values)

            # Accumulate losses
            losses['node'].append(t_losses[0])
            losses['path'].append(t_losses[1])
            losses['combined'].append(t_losses[2])
            losses['total'].append(t_losses[3])
            losses['consensus'].append(t_losses[4])

            # Accumulate Predictions
            for i, k in enumerate(predictions.keys()):
                pred_labels = np.zeros([self.config.data_sets._len_labels], dtype=np.int32)
                pred_labels[np.argmax(t_pred_probs[i])] = 1
                predictions[k].append(pred_labels.copy())
            targets.append(np.squeeze(target_label))

            if train is not None:
                # get the absolute maximum gradient to each variable
                gradients.append([np.max(np.abs(item)) for item in grads])

            if train and (step % self.config.batch_size == 0 or step == total_steps):
                # Update gradients after batch_size or at the end of the current epoch

                batch_size =  self.config.batch_size
                if step == total_steps:
                    batch_size = step%batch_size
                feed_dict[self.ph_batch_size] = batch_size

                sess.run([self.update_op], feed_dict=feed_dict)
                sess.run([self.reset_grads], feed_dict=feed_dict)

                if verbose and self.config.solver.gradients:
                    print("%d/%d :: " % (step, total_steps), end="")
                    for var, val in zip(['-'.join(k.name.split('/')[-2:]) for k in tf.trainable_variables()],
                                        np.mean(gradients, axis=0)):
                        print("%s :: %.8f  " % (var, val / self.config.batch_size), end="")
                    print("\n")
                sys.stdout.flush()

        # Average statistics over batches
        for k in losses.keys():
            losses[k] = np.mean(losses[k])
        for k in metrics.keys():
            metrics[k] = perf.evaluate(np.asarray(predictions[k]), np.asarray(targets), 0)

        coord.request_stop()
        coord.join(threads)
        return node_ids, predictions, losses, metrics, np.asarray(attn_values)

    def fit(self, sess, summary_writers):
        patience = self.config.patience
        learning_rate = self.config.solver.learning_rate

        inner_epoch, best_epoch, best_val_loss = 0, 0, 1e6
        nodes = {'train': None, 'val': None, 'test': None}
        losses = {'train': None, 'val': None, 'test': None}
        metrics = {'train': None, 'val': None, 'test': None}
        attn_values = {'train': None, 'val': None, 'test': None}
        predictions = {'train': None, 'val': None, 'test': None}
        best_losses, best_metrics, best_predictions = deepcopy(losses), deepcopy(metrics), deepcopy(predictions)

        while inner_epoch < self.config.max_inner_epochs:
            inner_epoch += 1
            nodes['train'], predictions['train'], losses['train'], metrics['train'], attn_values['train'] = \
                self.run_epoch(sess, data='train', train_op=self.accumulate_op, summary_writer=summary_writers['train'],
                               learning_rate=learning_rate)

            if inner_epoch % self.config.val_epochs_freq == 0:
                nodes['val'], predictions['val'], losses['val'], metrics['val'], attn_values['val'] = \
                    self.run_epoch(sess, data='val', train_op=None, summary_writer=summary_writers['val'], verbose=0)

                if self.config.run_test:
                    nodes['test'], predictions['test'], losses['test'], metrics['test'], attn_values['test'] = \
                        self.run_epoch(sess, data='test', train_op=None, summary_writer=summary_writers['test'], verbose=0)
                    self.print_inner_loop_stats(inner_epoch, metrics, losses)
                else:
                    print('---------- Epoch %d: tr_loss = %.2f val_loss %.2f || tr_micro = %.2f, val_micro = %.2f || '
                          'tr_acc = %.2f, val_acc = %.2f ' %
                          (inner_epoch, losses['train']['combined'], losses['val']['combined'],
                           metrics['train']['combined']['micro_f1'], metrics['val']['combined']['micro_f1'],
                           metrics['train']['combined']['accuracy'], metrics['val']['combined']['accuracy']))

                new_val_loss = losses['val']['node'] + losses['val']['combined'] + losses['val']['path']

            if new_val_loss < best_val_loss:
                if new_val_loss < (best_val_loss * self.config.improvement_threshold):
                    self.patience = self.config.patience
                    best_epoch = inner_epoch
                    best_losses = losses
                    best_metrics = metrics
                    best_predictions = predictions
                    self.saver.save(sess, self.config.ckpt_dir + 'inner-last-best')

                best_val_loss = new_val_loss
            else:
                if patience < 1:
                    # Restore the best parameters
                    self.saver.restore(sess, self.config.ckpt_dir + 'inner-last-best')
                    if learning_rate <= 0.00001:
                        print('Stopping by patience method')
                        break
                    else:
                        learning_rate /= 10
                        patience = self.config.patience
                        print('Learning rate dropped to %.8f' % learning_rate)
                else:
                    patience -= 1
        print('Best epoch: ', best_epoch)

        # Run Test set
        if not self.config.run_test:
            nodes['test'], best_predictions['test'], losses['test'], best_metrics['test'], attn_values['test'] = \
                self.run_epoch(sess, data='test', train_op=None, summary_writer=summary_writers['test'], verbose=0)

        # UPDATE LABEL CACHE
        self.dataset.update_label_cache('train', best_predictions['train']['combined'], ids=nodes['train'])
        self.dataset.update_label_cache('val', best_predictions['val']['combined'], ids=nodes['val'])
        self.dataset.update_label_cache('test', best_predictions['test']['combined'], ids=nodes['test'])

        return inner_epoch, nodes, best_losses, best_metrics, attn_values

    def fit_outer(self, sess, summary_writers):

        stats = []
        outer_epoch = 1
        flag = self.config.boot_reset
        patience = 1
        metrics = {'train': None, 'val': None, 'test': None}
        best_val_loss, best_metrics, best_attn_values = 1e6, None, None

        while outer_epoch <= self.config.max_outer_epochs:
            print('OUTER_EPOCH: ', outer_epoch)
            if outer_epoch == 2 and flag:  # reset after first bootstrap | Shall we reuse the weights ???
                print("------ Graph Reset | First bootstrap done -----")
                sess.run(self.init)  # reset all weights
                flag = False

            # Just to monitor the trainable variables in tf graph
            # print([v.name for v in tf.trainable_variables()], "\n")

            start = time.time()
            # Fit the model to predict best possible labels given the current estimates of unlabeled values
            inner_epoch, nodes, losses, metrics, attn_values = self.fit(sess, summary_writers)
            duration = time.time() - start
            stats.append(
                np.round([outer_epoch, inner_epoch,
                          losses['train']['combined'], losses['val']['combined'], losses['test']['combined'],
                          metrics['train']['combined']['micro_f1'], metrics['val']['combined']['micro_f1'], metrics['test']['combined']['micro_f1'],
                          metrics['train']['combined']['accuracy'], metrics['val']['combined']['accuracy'], metrics['test']['combined']['accuracy'],
                          duration], decimals=3))

            print('Outer Epoch %d: tr_loss = %.2f, val_loss %.3f te_loss %.3f|| '
                  'tr_micro = %.2f, val_micro = %.2f te_micro = %.3f|| '
                  'tr_acc = %.2f, val_acc = %.2f  te_acc = %.3f (%.3f sec)' %
                  (inner_epoch, losses['train']['combined'], losses['val']['combined'], losses['test']['combined'],
                   metrics['train']['combined']['micro_f1'], metrics['val']['combined']['micro_f1'], metrics['test']['combined']['micro_f1'],
                   metrics['train']['combined']['accuracy'], metrics['val']['combined']['accuracy'], metrics['test']['combined']['accuracy'],
                   duration))

            new_val_loss = losses['val']['combined'] + losses['train']['combined']
            if patience >= 1 and (new_val_loss < best_val_loss):
                if new_val_loss < (best_val_loss * self.config.improvement_threshold):
                    patience = 2
                    best_metrics = metrics
                    best_attn_values = attn_values
                best_val_loss = new_val_loss
            else:
                patience -= 1
                if patience < 1:
                    break
            outer_epoch += 1

        headers = ['Epoch', 'I_Epoch', 'TR_LOSS', 'VAL_LOSS', 'TE_LOSS', 'TR_MICRO', 'VAL_MACRO', 'TE_MACRO',
                   'TR_ACC', 'VAL_ACC', 'TE_ACC', 'DURATION']
        stats = tabulate(stats, headers)
        print(stats)
        print('Best Test Results || Accuracy %.3f | MICRO %.3f | MACRO %.3f' %
              (metrics['test']['combined']['accuracy'], metrics['test']['combined']['micro_f1'], metrics['test']['combined']['macro_f1']))
        return stats, nodes, best_metrics, best_attn_values

    def print_inner_loop_stats(self, inner_epoch, metrics, losses):

        print('---------- Epoch %d: tr_loss = %.2f val_loss %.2f te_loss %.2f ||'
              ' tr_micro = %.2f, val_micro = %.2f te_micro = %.2f|| '
              'tr_acc = %.2f, val_acc = %.2f  te_acc = %.2f ' %
              (inner_epoch, losses['train']['combined'], losses['val']['combined'], losses['test']['combined'],
               metrics['train']['combined']['micro_f1'], metrics['val']['combined']['micro_f1'], metrics['test']['combined']['micro_f1'],
               metrics['train']['combined']['accuracy'], metrics['val']['combined']['accuracy'], metrics['test']['combined']['accuracy']))

        print('########################################################################################')
        print('#~~~~~~~~~~~~~~~~~~~ tr_node_loss = %.2f val_node_loss %.2f te_node_loss %.2f ||\n'
              '#~~~~~~~~~~~~~~~~~~~ tr_path_loss = %.2f val_path_loss %.2f te_path_loss %.2f ||\n'
              '#~~~~~~~~~~~~~~~~~~~ tr_comb_loss = %.2f val_comb_loss %.2f te_comb_loss %.2f ||\n'
              '#~~~~~~~~~~~~~~~~~~~ tr_consensus_loss = %.2f val_consensus_loss %.2f te_consensus_loss %.2f ||\n'
              '#~~~~~~~~~~~~~~~~~~~ tr_total_loss = %.2f val_total_loss %.2f te_total_loss %.2f' %
              (losses['train']['consensus'], losses['val']['consensus'], losses['test']['consensus'],
               losses['train']['node'], losses['val']['node'], losses['test']['node'],
               losses['train']['path'], losses['val']['path'], losses['test']['path'],
               losses['train']['combined'], losses['val']['combined'], losses['test']['combined'],
               losses['train']['total'], losses['val']['total'], losses['test']['total']))

        print('########################################################################################')
        print('#~~~~~~~~~~~~~~~~~~~ tr_node_acc %.2f val_node_acc %.2f te_node_acc %.2f ||\n'
              '#~~~~~~~~~~~~~~~~~~~ tr_path_acc %.2f val_path_acc %.2f te_path_acc %.2f ||\n'
              '#~~~~~~~~~~~~~~~~~~~ tr_comb_acc %.2f val_comb_acc %.2f te_comb_acc %.2f ' %
              (metrics['train']['node']['accuracy'], metrics['val']['node']['accuracy'], metrics['test']['node']['accuracy'],
               metrics['train']['path']['accuracy'], metrics['val']['path']['accuracy'], metrics['test']['path']['accuracy'],
               metrics['train']['combined']['accuracy'], metrics['val']['combined']['accuracy'], metrics['test']['combined']['accuracy']))


def init_model(config):
    tf.reset_default_graph()
    tf.set_random_seed(1234)
    with tf.variable_scope('DEEP_DOPE', reuse=None) as scope:
        model = DeepDOPE(config)

    tf_config = tf.ConfigProto(allow_soft_placement=True)
    tf_config.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    if config.retrain:
        print("Loading model from checkpoint")
        load_ckpt_dir = config.ckpt_dir
    else:
        print("No model loaded from checkpoint")
        load_ckpt_dir = ''
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir, config=tf_config)
    return model, sess


def train_model(cfg):

    print('############## Training Module ')
    config = deepcopy(cfg)
    model, sess = init_model(config)
    with sess:
        summary_writers = model.add_summaries(sess)
        stats, nodes, test_metrics, attn_values = model.fit_outer(sess, summary_writers)
        return stats, nodes, test_metrics, attn_values


def main():

    args = Parser().get_parser().parse_args()
    print("=====Configurations=====\n", args)
    cfg = Config(args)
    train_percents = args.percents.split('_')
    folds = args.folds.split('_')

    outer_loop_stats = {}
    attention = {}
    results = {}
    nodes = {}

    #Create Main directories
    path_prefixes = [cfg.dataset_name, cfg.folder_suffix, cfg.data_sets.label_type]
    utils.create_directory_tree(path_prefixes)

    for train_percent in train_percents:
        cfg.train_percent = train_percent
        path_prefix = path.join(path.join(*path_prefixes), cfg.train_percent)
        utils.check_n_create(path_prefix)

        attention[train_percent] = {}
        results[train_percent] = {}
        outer_loop_stats[train_percent] = {}
        nodes[train_percent] = {}

        for fold in folds:
            print('Training percent: ', train_percent, ' Fold: ', fold, '---Running')
            cfg.train_fold = fold
            utils.check_n_create(path.join(path_prefix, cfg.train_fold))
            cfg.create_directories(path.join(path_prefix, cfg.train_fold))
            outer_loop_stats[train_percent][fold], nodes[train_percent][fold], results[train_percent][fold], \
                attention[train_percent][fold] = train_model(cfg)
            print('Training percent: ', train_percent, ' Fold: ', fold, '---completed')

    path_prefixes = [cfg.dataset_name, cfg.folder_suffix, cfg.data_sets.label_type]
    np.save(path.join(*path_prefixes, 'nodes.npy'), nodes)
    np.save(path.join(*path_prefixes, 'results.npy'), results)
    np.save(path.join(*path_prefixes, 'attentions.npy'), attention)
    np.save(path.join(*path_prefixes, 'outer_loop_stats.npy'), outer_loop_stats)

if __name__ == "__main__":
    np.random.seed(1234)
    main()
