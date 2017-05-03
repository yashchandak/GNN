from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf

from utils import *
from models import GCN, MLP
import Eval_Calculate_Performance as perf
import os

from sklearn.metrics import f1_score, recall_score, precision_score, accuracy_score
# Set random seed
seed = 123
np.random.seed(seed)

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset', 'cora', 'Dataset string.')  # 'cora', 'citeseer', 'pubmed'
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, '0.01 Initial learning rate.')
flags.DEFINE_integer('epochs', 5000, 'Number of epochs to train.')
flags.DEFINE_integer('hidden1', 16, '16 Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, '0.5 Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, '5e-4 Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 3, 'Maximum Chebyshev polynomial degree.')


def main(path='', percent=1, fold=1, label_type='labels'):
    # Initialize session
    tf.reset_default_graph()
    tf.set_random_seed(seed)

    tfconfig = tf.ConfigProto(allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sess = tf.Session(config=tfconfig)

    # Load data
    adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(path, percent, fold, label_type)
    #adj, features, y_train, y_val, y_test, train_mask, val_mask, test_mask = load_data(FLAGS.dataset)

    # Some preprocessing
    features = preprocess_features(features)
    if FLAGS.model == 'gcn':
        support = [preprocess_adj(adj)]
        num_supports = 1
        model_func = GCN
    elif FLAGS.model == 'gcn_cheby':
        support = chebyshev_polynomials(adj, FLAGS.max_degree)
        num_supports = 1 + FLAGS.max_degree
        model_func = GCN
    elif FLAGS.model == 'dense':
        support = [preprocess_adj(adj)]  # Not used
        num_supports = 1
        model_func = MLP
    else:
        raise ValueError('Invalid argument for model: ' + str(FLAGS.model))

    # Define placeholders
    placeholders = {
        'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
        'features': tf.sparse_placeholder(tf.float32, shape=tf.constant(features[2], dtype=tf.int64)),
        'labels': tf.placeholder(tf.float32, shape=(None, y_train.shape[1])),
        'labels_mask': tf.placeholder(tf.int32),
        'dropout': tf.placeholder_with_default(0., shape=()),
        'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
    }

    # Create model
    model = model_func(placeholders, input_dim=features[2][1], logging=True)

    # Define model evaluation function
    def evaluate(features, support, labels, mask, placeholders, metrics=True):
        t_test = time.time()
        feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
        outs_val = sess.run([model.loss, model.outputs], feed_dict=feed_dict_val)

        acc = accuracy_score(np.argmax(labels, axis=1), np.argmax(outs_val[1], axis=1), sample_weight=mask)

        labels, preds = labels, outs_val[1]
        labels_masked, preds_masked = [],[]
        for idx, val in enumerate(mask):
            if val:
                labels_masked.append(labels[idx])
                preds_masked.append(preds[idx])
        if metrics:
            mets = perf.evaluate(preds_masked, labels_masked)
        else:
            mets = [0]*20
        return outs_val[0], (time.time() - t_test), mets, acc , preds


    # Init variables
    sess.run(tf.global_variables_initializer())

    cost_val = []

    # Train model
    for epoch in range(FLAGS.epochs):

        t = time.time()
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss], feed_dict=feed_dict)

        # Validation
        #_, __, metrics_train, acc_train = evaluate(features, support, y_train, train_mask, placeholders, metrics=False)
        cost, duration, metrics, acc_val, preds = evaluate(features, support, y_val, val_mask, placeholders, metrics=False)
        cost_val.append(cost)
        # Print results

        print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
               "val_loss=", "{:.5f}".format(cost),
              "val_acc=", "{:.5f}".format(acc_val), "time=", "{:.5f}".format(time.time() - t))

        # Print results
        #print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
        #      "train_acc=", "{:.5f}".format(0), "val_loss=", "{:.5f}".format(cost),
        #      "val_acc=", "{:.5f}".format(metrics[-1]), "time=", "{:.5f} \n".format(time.time() - t), metrics, 0, acc_val)

        if epoch > FLAGS.early_stopping and cost_val[-1] > np.mean(cost_val[-(FLAGS.early_stopping+1):-1]):
            print("Early stopping...")
            break

    print("Optimization Finished!")

    # Testing
    test_cost, test_duration, metrics, acc_test, preds  = evaluate(features, support, y_test, test_mask, placeholders)
    print("Test set results:", "cost=", "{:.5f}".format(test_cost), "time=", "{:.5f}\n".format(test_duration), metrics, acc_test)
    return metrics, preds



def write_results(all_results, path ='./'):
    for percent, results in all_results.items():
        f = open(path + str(percent) + '.txt', 'w')

        for metric in metric_vars:
            f.write(metric + '\t')
        f.write('\n')

        arr = np.zeros((len(results.values()[0]), len(results)))  # [[]]*len(results.values()[0])
        for shuff, vals in results.items():
            for idx, val in enumerate(vals):
                arr[idx][shuff - 1] = val
                f.write(str(val) + '\t')
            f.write('\n')

            # f.write('\n')
        for v in range(arr.shape[0]):
            f.write(str(np.mean(arr[v][:])) + '\t')

        f.write('\n')
        for v in range(arr.shape[0]):
            f.write(str(np.var(arr[v][:])) + '\t')

        f.close()


metric_vars = ['coverage','average_precision','ranking_loss','micro_f1','macro_f1','micro_precision',
               'macro_precision','micro_recall','macro_recall','p@1','p@3','p@5','hamming_loss','bae','cross-entropy','accuracy']
path = '/home/priyesh/Desktop/Codes/Sample_Run/Datasets/'

#main(path, 1,1)
dataset = 'citeseer/'
label_type = 'labels_dfs'
percents = [1,2,3,4,5,6]
#percents = [1,2,3,4,5,6]
folds =  np.arange(1,6)
#percents = [20]
#folds = [1,2,3]

for per in percents:
    print("\n\n ----- Percent: {} -----\n".format(per))
    all_results = {per:{}}
    p = './'+dataset+'/'+str(per)+'/'
    if not os.path.exists(p): os.makedirs(p)
    for f in folds:
        all_results[per][f], preds = main(path+dataset, per, f, label_type=label_type)
        write_results(all_results, p)
        np.save(p+'labels-'+str(per)+'-'+str(f), preds)
        #exit()
