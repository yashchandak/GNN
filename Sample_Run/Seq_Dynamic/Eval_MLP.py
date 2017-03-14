from __future__ import print_function

import tensorflow as tf
import numpy as np
import time, sys

from Eval_Data import Data
from Eval_Config import Config
import Eval_Calculate_Performance as perf
import Eval_utils as utils



class Network:
    def __init__(self, cfg):
        self.cfg = cfg

    def loss(self, y_pred, y):
        #cross_loss = tf.add(tf.log(1e-10 + y_pred)*y, tf.log(1e-10 + (1-y_pred))*(1-y))
        #cross_entropy = -1*tf.reduce_mean(tf.reduce_sum(cross_loss,1))

        ##elf.label_sigmoid = tf.nn.softmax(predictions)
        #cross_entropy = tf.reduce_mean(-tf.reduce_sum(y * tf.log(y_pred + 1e-10), 1))
        cross_entropy = tf.reduce_mean(tf.squared_difference(y_pred, y))

        vars   = tf.trainable_variables() 
        lossL2 = tf.add_n([tf.nn.l2_loss(v) for v in vars])*0.0000
    
        total_loss = cross_entropy# + lossL2
        return total_loss

    def weight_variable(self, name,  shape):
        initial = tf.truncated_normal(shape, stddev=1/shape[0])
        return tf.Variable(initial, name=name)

    def bias_variable(self, name, shape):
        initial = tf.constant(0.01, shape=shape)
        return tf.Variable(initial, name=name)

    def prediction(self, x, keep_prob):
        x = tf.nn.dropout(x, keep_prob)
        with tf.variable_scope('Network'):
            if self.cfg.hidden > 0:
                W1 = self.weight_variable('weight1', [self.cfg.input_len, self.cfg.hidden])
                b1 = self.bias_variable( 'bias1', [self.cfg.hidden])
                
                W2 = self.weight_variable('weight2', [self.cfg.hidden, self.cfg.label_len] )
                b2 = self.bias_variable( 'bias2', [self.cfg.label_len])
            
                hidden = tf.tanh(tf.matmul(x, W1) + b1)
                hidden_drop = tf.nn.dropout(hidden, keep_prob)
                
                self.y_pred = tf.nn.softmax(tf.matmul(hidden_drop, W2) + b2)

            else:
                W1 = self.weight_variable('weight1',[self.cfg.input_len, self.cfg.label_len] )
                b1 = self.bias_variable( 'bias1', [self.cfg.label_len])

                self.y_pred = tf.nn.softmax(tf.matmul(x, W1) + b1)

        return self.y_pred
    
        
    def train(self, loss, optimizer):
        #tf.train.AdamOptimizer(cfg.learning_rate)
        train_step = optimizer.minimize(loss)
        return train_step


class Model:
    def __init__(self, config):
        self.config = config
        self.data   = Data(config)
        self.add_placeholders()

        self.net      = Network(self.config)
        self.y_pred   = self.net.prediction(self.x, self.keep_prob)
        
        self.optimizer= self.config.optimizer
        self.loss     = self.net.loss(self.y_pred, self.y)
        self.train    = self.net.train(self.loss, self.optimizer)

        self.saver = tf.train.Saver()
        self.init = tf.global_variables_initializer()

    def add_placeholders(self):
        self.x = tf.placeholder(tf.float32, shape=[None, self.config.input_len])
        self.y = tf.placeholder(tf.float32, shape=[None, self.config.label_len])
        self.keep_prob = tf.placeholder(tf.float32)

    def run_epoch(self, sess, data):
        err = []
        i = 0
        for inputs, labels, tot in self.data.next_batch(data):
            #Mini-batch execute
            feed_dict = {self.x:inputs, self.y:labels, self.keep_prob:self.config.drop}
            _,  loss, y_  = sess.run([self.train, self.loss, self.y_pred], feed_dict=feed_dict)
            err.append(loss)    
            sys.stdout.write("\rRun: {}/{}:: Loss = {}".format(i,tot,np.mean(err)))
            sys.stdout.flush()
            i += 1

        #metrics = perf.evaluate(y_, labels)
        #print("\nTraining error\n")
        #self.print_metrics(metrics)

        return np.mean(err)


    def run_eval(self, sess, data):
        #check Evaluation dataset
        y, y_preds = [],[]
        for inputs_valid, labels_valid, tot in self.data.next_batch(data=data):
            feed_dict = {self.x:inputs_valid, self.y:labels_valid, self.keep_prob:1}
            y_, loss = sess.run([self.y_pred, self.loss], feed_dict=feed_dict)

            y_preds.extend(y_)
            y.extend(labels_valid)
        metrics = perf.evaluate(y_preds, y, self.config.threshold)

        return loss, metrics


    def print_metrics(self, inp):
        for idx, item in enumerate(inp):
            print(self.config.metrics[idx], ": ", item)


    def fit(self, sess):
        #define parametrs for early stopping early stopping
        max_epochs = self.config.max_epochs
        patience = self.config.patience  # look as this many examples regardless
        patience_increase = self.config.patience_increase  # wait this much longer when a new best is found
        improvement_threshold = self.config.improvement_threshold  # a relative improvement of this much is considered significant

        validation_loss = 1e6
        step = 1
        best_step = -1
        losses = []
        learning_rate = self.config.learning_rate

        while step <= self.config.max_epochs:

            start_time = time.time()
            average_loss = self.run_epoch(sess, data='train')
            duration = time.time() - start_time


            if (step % self.config.val_epochs_freq == 0):
                val_loss, metrics = self.run_eval(sess, data='val')

                sys.stdout.write('\n Epoch %d: tr_loss = %.2f, val_loss = %.2f  (%.3f sec)'% (step, average_loss, val_loss, duration))
                sys.stdout.flush()	
                # Save model only if the improvement is significant
                if (val_loss < validation_loss * improvement_threshold):
                    #patience = max(patience, step * patience_increase)
                    validation_loss = val_loss
                    best_step = step
                    patience = step + max(self.config.val_epochs_freq,self.config.patience_increase)
                    print('best step %d' % (best_step))
                    self.saver.save(sess, self.config.directory+'last-best')
		
                elif val_loss > validation_loss * improvement_threshold:
                    patience = step - 1

            else:
                # Print status to stdout.
                sys.stdout.write('Epoch %d: loss = %.2f  (%.3f sec)' % (step, average_loss, duration))
                sys.stdout.flush()
                
            if (patience <= step):
                #config.val_epochs_freq = 2
                learning_rate = learning_rate / 10
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                patience = step + max(self.config.val_epochs_freq,self.config.patience_increase)
                print('--------- Learning rate dropped to: %f'%(learning_rate))		
                if learning_rate <= 0.0000001:
                    print('Stopping by patience method')
                    break

            losses.append(average_loss) 
            step += 1

        print("Test Results") #[IMP]replace with Test data
        #Reload the best state
        tfconfig = tf.ConfigProto(allow_soft_placement=True)
        tfconfig.gpu_options.allow_growth = True
        sess = tf.Session(config = tfconfig)
        new_saver = tf.train.import_meta_graph(self.config.directory+'last-best.meta')
        new_saver.restore(sess, tf.train.latest_checkpoint(self.config.directory))

        test_loss, metrics = self.run_eval(sess, data='test')
        self.print_metrics(metrics) 

        return losses, best_step, metrics




def evaluate(cfg):
    #with tf.variable_scope('Evaluation', reuse=None) as scope:
    print("=====Configurations=====\n", cfg.__dict__)
    model = Model(cfg)
    
    tfconfig = tf.ConfigProto()
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config = tfconfig)
    
    all_results = {}
    for train_percent in cfg.training_percents:
        all_results[train_percent] = {}
        for shuf in cfg.num_shuffles:
             sess.run(model.init)
             model.data.set_training_validation(train_percent,shuf)
             losses, best_step, metrics = model.fit(sess)
             all_results[train_percent][shuf] = metrics

    for train_percent in sorted(all_results.keys()):
        print ('Train percent:', train_percent)
        micro, accuracy = [], []
        #print numpy.mean(all_results[train_percent])
        x = all_results[train_percent]
        for v in x.values():
            micro.append(v[3])
            accuracy.append(v[-1])
        print (x.values())
        print ("Micro: ",np.mean(micro), "  Accuracy: ",np.mean(accuracy))
        print ('-------------------') 
        utils.write_results(cfg, all_results)
 


def evaluate_multi():
    cfg = Config()
    meta_param = {'learning_rate': [0.001],
                  'max_epochs': [1]
    }

    variations = len(meta_param['learning_rate'])
    
    #Make sure number of variants are equal
    for k,v in meta_param.items():
        assert len(v) == variations           
    
    for idx in range(variations):        
        cfg.create("Results-"+str(idx))
        for k,vals in meta_param.items():
            setattr(cfg, k, vals[idx])
            #write config details to a file
            print(k, vals[idx])

        #All set... GO!
        cfg.init2()
        evaluate(cfg)
        print('\n\n ===================== \n\n')



if __name__ == "__main__":
    con  = Config('cora/','/home/priyesh/Desktop/Codes/Sample_Run/Datasets/cora/features.npy')
    #Embeddings and checkpoint directorty
    with tf.device('/gpu:1'):
        evaluate(con)

