from __future__ import generators, print_function
import numpy as np
import tensorflow as tf
import threading
from random import shuffle
from scipy.io import loadmat


class DataSet(object):
    def __init__(self, cfg):
        """Construct a DataSet.
        """
        self.cfg = cfg
        self.all_walks   = np.fliplr(np.loadtxt(cfg.walks_dir, dtype=np.int))  # reverse the sequence
        self.node_seq    = self.all_walks[:, -1]  # index by ending node
        self.all_labels  = self.get_labels(cfg.label_dir)
        self.all_features= self.get_fetaures(cfg.features_dir)

        #Increment the positions by 1 and mark the 0th one as False
        self.train_nodes = np.concatenate(([False], np.load(cfg.label_fold_dir + 'train_ids.npy')))
        self.val_nodes   = np.concatenate(([False], np.load(cfg.label_fold_dir + 'val_ids.npy')))
        self.test_nodes  = np.concatenate(([False], np.load(cfg.label_fold_dir + 'test_ids.npy')))
        # [!!!IMP!!]Assert no overlap between test/val/train nodes

        self.change = 0
        self.label_cache, self.update_cache = {0:list(self.all_labels[0])}, {}

    def get_fetaures(self, path):
        # Serves 2 purpose:
        # a) add feature for dummy node 0 a.k.a <EOS> and <unlabeled>
        # b) increments index of all features by 1, thus aligning it with indices in walks
        all_features = np.load(path)
        all_features = all_features.astype(np.float32, copy=False)  # Required conversion for Python3
        all_features = np.concatenate(([np.zeros(all_features.shape[1])], all_features), 0)
        return all_features

    def get_labels(self, path):
        # Labels start with node '0'; Walks_data with node '1'
        # To get corresponding mapping, increment the label node number by 1
        # add label for dummy node 0 a.k.a <EOS> and <unlabeled>
        all_labels = np.load(path)
        all_labels = np.concatenate(([np.zeros(all_labels.shape[1])], all_labels), 0)

        return all_labels

    def accumulate_label_cache(self, labels, nodes):
        #Aggregates all the labels for the corresponding nodes
        #and tracks the count of updates made
        default = (self.all_labels[0], 0) #Initial estimate -> all_zeros
        #WTF!labels = labels[0]
        
        if self.cfg.data_sets.binary_label_updates:
            #Convert to binary and keep only the maximum value as 1
            amax = np.argmax(labels, axis = 1)
            labels = np.zeros(labels.shape)
            for idx, pos in enumerate(amax):
                labels[idx,pos] = 1
        
        for idx, node in enumerate(nodes):
            prv_label, prv_count = self.update_cache.get(node, default)
            new_label = prv_label + labels[idx]
            new_count = prv_count + 1
            self.update_cache[node] = (new_label, new_count)

    def update_label_cache(self):
        #Average all the predictions made for the corresponding nodes and reset cache
        alpha = self.cfg.solver.label_update_rate

        if len(self.label_cache.items()) <= 1: alpha =1

        for k, v in self.update_cache.items():
            old = self.label_cache.get(k, self.label_cache[0])
            new = (1-alpha)*np.array(old) + alpha*(v[0]/v[1])
            self.change += np.mean((new - old) **2)
            self.label_cache[k] =  list(new)

        print("\nChange in label: :", np.sqrt(self.change/self.cfg.data_sets._len_vocab)*100)
        self.change = 0
        self.update_cache = {}

    def get_nodes(self, dataset):
        nodes = []
        if dataset == 'train':
            nodes = self.train_nodes
        elif dataset == 'val':
            nodes = self.val_nodes
        elif dataset == 'test':
            nodes = self.test_nodes
        elif dataset == 'all':
            # Get all the nodes except the 0th node
            nodes = [True]*len(self.train_nodes)
            nodes[0] = False
        else:
            raise ValueError

        return nodes

    def next_batch(self, dataset, batch_size, shuffle=True):
        ctr = 0

        nodes = self.get_nodes(dataset)
        label_len = np.shape(self.all_labels)[1]

        # Get position of all walks ending with desired set of nodes
        pos = []
        for node in np.where(nodes)[0]:
            pos.extend(np.where(self.node_seq == node)[0])

        pos = np.array(pos)
        if shuffle:
            indices = np.random.permutation(len(pos))
            pos = pos[indices]

        if batch_size == -1:
            batch_size = len(pos)

        tot = len(pos)//batch_size
        for i in range(0, len(pos), batch_size):
            x = self.all_walks[pos[i: i + batch_size]]
            x = np.swapaxes(x, 0, 1) # convert from (batch x step) to (step x batch)

            # get labels for valid data points, for others: select the 0th label
            x2 = [[self.label_cache.get(item, list(self.all_labels[0])) for item in row] for row in x]
            y  = [list(self.all_labels[item]) for item in x[-1]]

            # get features for all data points
            x = [[self.all_features[item] for item in row] for row in x]

            seq = self.node_seq[pos[i: i + batch_size]]
            ctr += 1
            yield (x, x2, seq, y, tot, ctr)


    def next_batch_same(self, dataset, node_count=1):

        nodes = self.get_nodes(dataset)

        pos = []
        counts = []
        seq = []
        for node in np.where(nodes)[0]:
            temp = np.where(self.node_seq == node)[0]
            counts.append(len(temp))
            seq.append(node)
            pos.extend(temp)

        pos = np.array(pos)

        start = 0
        max_len = self.all_walks.shape[1]
        # Get a batch of all walks for 'node_count' number of node
        for idx in range(0, len(counts), node_count):
            #print(idx)
            stop = start + np.sum(counts[idx:idx+node_count]) #start + total number of walks to be consiudered this time
            x = self.all_walks[pos[start:stop]] #get the walks corresponding to respective positions

            temp = np.array(x)>0  #get locations of all zero inputs
            lengths = max_len - np.sum(temp, axis=1)

            x = np.swapaxes(x, 0, 1) # convert from (batch x step) to (step x batch)

            # get labels for valid data points, for others: select the 0th label
            x2 = [[self.label_cache.get(item, list(self.all_labels[0])) for item in row] for row in x]
            y  = [list(self.all_labels[item]) for item in x[-1,:]] #Not useful, only presetn for sake of placeholder

            # get features for all data points
            x1 = [[self.all_features[item] for item in row] for row in x]

            start = stop
            yield (x, x1, x2, seq[idx:idx+node_count], counts[idx:idx+node_count], y, lengths)



class Qrunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self, cfg, dataset):
        self.cfg = cfg
        self.dataset = dataset
        #self.data_iterator = data
        self.n_threads = 4
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.dataY = tf.placeholder(dtype=tf.int64, shape=[None, ])
        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.

        self.queue =  tf.RandomShuffleQueue(shapes=[[10], []],
                      dtypes=[tf.float32, tf.int64],
                      capacity=100,
                      min_after_dequeue=0)

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])

        #self.qr = tf.train.QueueRunner(self.queue, [self.enqueue_op] * 4)

    def data_iterator(self, n):
        #global flag
        # while True:
        print('===============TRAIN==========')
        total_samples = 1024
        samples_per_thread = total_samples//self.n_threads
        start = n*samples_per_thread
        for idx in range(start, start + samples_per_thread, batch_size):
            feats = np.random.randn(batch_size, 10)
            labels = np.random.randint(0, 2, batch_size)
            yield feats, labels
        return

    def data_iterator2(self, n):
        #global flag
        # while True:
        print('===============VAL==========')
        total_samples = 1024
        #define total number of batches
        samples_per_thread = total_samples//self.n_threads
        start = n*samples_per_thread
        for idx in range(start, start + samples_per_thread, batch_size):
            feats = np.ones((batch_size, 10))
            labels = np.random.randint(0, 2, batch_size)
            yield feats, labels
        return

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        images_batch, labels_batch = self.queue.dequeue_many(batch_size)
        return images_batch, labels_batch

    def thread_main(self, sess, coord, n, samples='train'):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        if samples == 'train': fn = self.data_iterator
        elif samples == 'val': fn = self.data_iterator2

        for dataX, dataY in fn(n):
            sess.run(self.enqueue_op, feed_dict={self.dataX:dataX, self.dataY:dataY})

        #exit()
        #coord.request_stop() #Non-deterministic, can make other threads stop early
        #sess.run(self.queue.close(cancel_pending_enqueues=False))

    def start_threads(self, sess, coord, samples):
        """ Start background threads to feed queue """
        threads = []
        for n in range(self.n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess, coord, n, samples))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads



# simple model
w = tf.get_variable("w1", [10, 2])
y_pred = tf.matmul(images_batch, w)
sum = tf.reduce_sum(images_batch)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, labels_batch)

# for monitoring
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
init = tf.initialize_all_variables()
sess.run(init)

def step(samples):
    coord = tf.train.Coordinator()
    # start the tensorflow QueueRunner's
    #threads1 = tf.train.start_queue_runners(sess=sess, coord=coord)
    # start our custom queue runner's threads
    threads = custom_runner.start_threads(sess, coord, samples=samples)
    print "%d Threads started"%len(threads)

    ctr = 0
    #One epoch
    try:
        while True :
            _, sum_val, loss_val = sess.run([train_op, sum, loss_mean])
            print "ctr %d : "%ctr, loss_val, sum_val, sess.run(custom_runner.queue.size()), threading.activeCount()
            ctr += 1

            #if coord.should_stop() and sess.run(custom_runner.queue.size()) == 0: break
            if threading.active_count() == 1  and sess.run(custom_runner.queue.size()) == 0: break

    except Exception, e:
        print('catched exception: ', e)
        coord.request_stop(e)
    #finally:
    #coord.request_stop()
    #coord.join(threads)
