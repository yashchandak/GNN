from __future__ import generators, print_function
import numpy as np
from random import shuffle
from scipy.io import loadmat

import functools
import Queue
#from multiprocessing import Process, Queue, Manager, Pool
import threading
import time
from collections import defaultdict


def async_prefetch_wrapper(iterable, buffer=100):
    """
    wraps an iterater such that it produces items in the background
    uses a bounded queue to limit memory consumption
    """
    done = 'DONE'# object()

    def worker(q, it):
        for item in it:
            q.put(item)
        q.put(done)

    # launch a thread to fetch the items in the background
    queue = Queue.Queue(buffer)

    #pool = Pool()
    #m = Manager()
    #queue = m.Queue()
    it = iter(iterable)
    #workers = pool.apply_async(worker, (queue, it))
    thread = threading.Thread(target=worker, args=(queue, it))
    #thread = Process(target=worker, args=(queue, it))
    thread.daemon = True
    thread.start()
    # pull the items of the queue as requested
    while True:
        item = queue.get()
        if item == 'DONE':#done:
            return
        else:
            yield item

    #pool.close()
    #pool.join()


def async_prefetch(func):
    """
    decorator to make generator functions fetch items in the background
    """
    @functools.wraps(func)
    def wrapper(*args, **kwds):
        return async_prefetch_wrapper(func(*args, **kwds))

    return wrapper

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
        self.label_cache, self.update_cache = {0:self.all_labels[0]}, {}
        self.wce = self.get_wce()

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

    def get_wce(self):
        if self.cfg.solver.wce:
            valid = self.train_nodes + self.val_nodes
            tot = np.dot(valid, self.all_labels)
            wce = 1 - tot*1.0/np.sum(tot)
        else:
            wce = [1]*self.all_labels.shape[1]

        print("Cross-Entropy weights: ",wce)
        return wce


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
            new = (1-alpha)*old + alpha*(v[0]/v[1])
            self.change += np.mean((new - old) **2)
            self.label_cache[k] =  new

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

    @async_prefetch
    def next_batch(self, dataset, batch_size, shuffle=True):

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
            x2 = [[self.label_cache.get(item, self.all_labels[0]) for item in row] for row in x]
            y  = [self.all_labels[item] for item in x[-1]]

            # get features for all data points
            x = [[self.all_features[item] for item in row] for row in x]

            seq = self.node_seq[pos[i: i + batch_size]]

            yield (x, x2, seq, y, tot)

    @async_prefetch
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

            #"""
            #original
            # get labels for valid data points, for others: select the 0th label
            x2 = [[self.label_cache.get(item, self.all_labels[0]) for item in row] for row in x]
            y  = [self.all_labels[item] for item in x[-1,:]] #Not useful, only presetn for sake of placeholder

            # get features for all data points
            x1 = [[self.all_features[item] for item in row] for row in x]
            #"""

            """
            #Unique based
            u, inv = np.unique(x, return_inverse=True)
            u2, inv2 = np.unique(x[-1:], return_inverse=True)
            x2 = np.array([self.label_cache.get(item, self.all_labels[0]) for item in u])[inv]#.reshape(x.shape)
            x1 = np.array([self.all_features[item] for item in u])[inv]#.reshape(x.shape)
            y = np.array([self.all_labels[item] for item in u2])[inv2]
            """

            """
            # Vectorized
            # get labels for valid data points, for others: select the 0th label
            x2 = np.vectorize(self.label_cache.get)(x)
            x1 = np.vectorize(self.all_features.__getitem__)(x)
            y = np.vectorize(self.all_labels.__getitem__)(x[-1:])
            """

            start = stop
            yield (x, x1, x2, seq[idx:idx+node_count], counts[idx:idx+node_count], y, lengths)


    def testPerformance(self):

        start = time.time()
        step =0
        for a,b,c,d,e,f,g in self.next_batch_same('all'):
            step += 1
            if step%500 == 0: print(step)

        print ('total time: ', time.time()-start)