from __future__ import generators, print_function
import numpy as np
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
        labels = labels[0]
        
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
        for k, v in self.update_cache.items():
            self.label_cache[k] = list(v[0]/v[1])  
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
            x2 = [[self.label_cache.get(item, self.label_cache[0]) for item in row] for row in x]
            y  = [list(self.all_labels[item]) for item in x[-1]]

            # get features for all data points
            x = [[self.all_features[item] for item in row] for row in x]

            seq = self.node_seq[pos[i: i + batch_size]]

            yield (x, x2, seq, y, tot)
