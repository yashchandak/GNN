from __future__ import generators, print_function
import numpy as np
import time
from copy import deepcopy
from os import path
from collections import defaultdict

class DataSet(object):
    def __init__(self, cfg):
        """Construct a DataSet.
        """
        self.cfg = cfg
        self.all_labels  = self.get_labels(cfg.label_dir)
        self.all_features= self.get_fetaures(cfg.features_dir)

        #Increment the positions by 1 and mark the 0th one as False
        self.train_nodes = np.concatenate(([False], np.load(cfg.label_fold_dir + 'train_ids.npy')))
        self.val_nodes   = np.concatenate(([False], np.load(cfg.label_fold_dir + 'val_ids.npy')))
        self.test_nodes  = np.concatenate(([False], np.load(cfg.label_fold_dir + 'test_ids.npy')))
        # [!!!IMP!!]Assert no overlap between test/val/train nodes

        self.change = 0
        self.path_pred_variance = {}
        self.label_cache, self.update_cache = {0:self.all_labels[0]}, {}
        self.wce = self.get_wce()

        self.dist, self.diameter, self.degree = self.load_graph()
        self.diameter = min(cfg.max_depth, self.diameter)

        self.n_train_nodes = np.sum(self.train_nodes)
        self.n_val_nodes = np.sum(self.val_nodes)
        self.n_test_nodes = np.sum(self.test_nodes)
        self.n_nodes = len(self.train_nodes)

        self.n_features = self.all_features.shape[1]
        self.n_labels = self.all_labels.shape[1]
        self.multi_label = self.is_multilabel()

        self.sampler_reset = False
        self.print_statistics()


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
            wce = 1/(len(tot) * (tot*1.0/np.sum(tot)))
        else:
            wce = [1]*self.all_labels.shape[1]

        print("Cross-Entropy weights: ",wce)
        return wce

    def load_graph(self):
        dist = np.load(open(self.cfg.dist_dir, 'rb'), encoding='latin1')
        diameter = np.max([depth for node, neigh in dist.items() for n, depth in neigh.items()])
        degree = [np.sum(np.array(list(dist[node_id].values()), dtype=np.int) == 1) for node_id in range(len(dist))]
        degree.insert(0, 0)
        return dist, diameter, np.array(degree)

    def get_degree(self, node_id):
        return self.degree[node_id]

    def is_multilabel(self):
        sum = np.count_nonzero(self.all_labels[self.train_nodes])
        return sum > self.n_train_nodes

    def print_statistics(self):
        print('############### DATASET STATISTICS ####################')
        print('Train Nodes: %d \nVal Nodes: %d \nTest Nodes: %d \nFeatures: %d \nLabels: %d \nMulti-label: %s \nDiameter: %d \nMax Degree: %d'\
              %(self.n_train_nodes, self.n_val_nodes, self.n_test_nodes, self.n_features, self.n_labels, self.multi_label, self.diameter, np.max(self.degree)))
        print('-----------------------------------------------------\n')

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

        update_no = len(self.path_pred_variance.items())
        self.path_pred_variance[update_no] = {}

        if len(self.label_cache.items()) <= 1: alpha =1

        for k, v in self.update_cache.items():
            old = self.label_cache.get(k, self.label_cache[0])
            cur = v[0]/v[1]
            new = (1-alpha)*old + alpha*cur
            self.change += np.mean((new - old) **2)
            self.path_pred_variance[update_no][k] = cur
            self.label_cache[k] =  new

        print("\nChange in label: :", np.sqrt(self.change/self.cfg.data_sets._len_vocab)*100)
        self.change = 0
        self.update_cache = {}

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

    def next_batch_same(self, dataset, node_count=1, shuffle=False):

        nodes = self.get_nodes(dataset)
        nodes = np.where(nodes)[0]
        if shuffle:
            indices = np.random.permutation(len(nodes))
            nodes  = nodes[indices]

        pos = []
        counts = []
        seq = []

        for node in nodes:
            temp = np.where(self.node_seq == node)[0]
            counts.append(len(temp))
            seq.append(node)
            pos.extend(temp)
        pos = np.array(pos)

        start = 0
        max_len = self.all_walks.shape[1]
        tot = len(nodes)//node_count
        # Get a batch of all walks for 'node_count' number of node
        for idx in range(0, len(counts), node_count):
            stop = start + np.sum(counts[idx:idx+node_count]) #start + total number of walks to be consiudered this time
            x = self.all_walks[pos[start:stop]] #get the walks corresponding to respective positions
            x = np.swapaxes(x, 0, 1) # convert from (batch x step) to (step x batch)

            temp = np.array(x)>0  #get locations of all zero inputs as binary matrix
            lengths = max_len - np.sum(temp, axis=0)

            x1 = [[self.all_features[item] for item in row] for row in x] # get features for all data points
            x2 = [[self.label_cache.get(item, self.all_labels[0]) for item in row] for row in x] # get pseudo labels
            y  = [self.all_labels[item] for item in x[-1,:]] #get tru labels for Node of interest

            start = stop
            yield (x, x1, x2, seq[idx:idx+node_count], counts[idx:idx+node_count], y, lengths, tot)

    def get_dataset_sequences(self, file):
        for line in open(path.join(self.data_dir, file)).readlines():
            yield line

    def next_sequence_loop(self, sequence_generator):
        #Never-ending generator
        gen = sequence_generator.__iter__()
        while True:
            for lines in gen:
                seq = []
                for line in lines:
                    seq.append([int(node_id) for node_id in line])
                yield np.asarray(seq, dtype=np.int32)
                #yield np.asarray(seq, dtype=np.int32).squeeze() #Don't squeeze that will help us count the n_walks
            gen = sequence_generator.__iter__() #Re-initiate hte generator

    def get_batches(self, batch_size, total_batches, generator):
        batch = []
        batch_cnt = 0
        n_batches = 0
        for seq in self.next_sequence_loop(generator):
            batch.append(seq)
            batch_cnt += 1
            if batch_cnt >= batch_size:
                res = deepcopy(batch)
                batch = []
                batch_cnt = 0
                n_batches += 1
                yield res

            if n_batches >= total_batches:
                break


    def sample_DFS_walks(self, node_id, n_walks=1, by_max_degree=False, by_prob=False, max_walks=5):
        # dist array starts from 0 :/
        walks = np.empty((0, self.diameter + 1), int)
        connected_nodes = np.array(list(self.dist[node_id - 1].keys()), dtype=np.int)
        connected_nodes_depth = np.array(list(self.dist[node_id - 1].values()), dtype=np.int)

        if by_max_degree:
            immediate_neighbors = connected_nodes[connected_nodes_depth == 1]
            additional_neighbors_reqd = max(0, max_walks - self.degree[node_id])
            p = None
            if immediate_neighbors.shape[0] == 0:
                path = [node_id-1]
                arr = np.array(path) + 1
                npad = (self.diameter + 1) - len(arr)
                if npad > 0:
                    arr = np.lib.pad(arr, (0, npad), 'constant', constant_values=(0))
                walks = np.append(walks, arr.reshape(1, self.diameter + 1), axis=0)
            else:
                if additional_neighbors_reqd:
                    immediate_neighbors = np.append(immediate_neighbors,
                                                    np.random.choice(immediate_neighbors,
                                                                     size=additional_neighbors_reqd,
                                                                     replace=True))
                for neighbor_id in immediate_neighbors:
                    curr_node = neighbor_id
                    path = [curr_node, node_id - 1]

                    while len(path) <= self.diameter:
                        cn_connected_nodes = np.array(list(self.dist[curr_node].keys()), dtype=np.int)
                        cn_connected_nodes_depth = np.array(list(self.dist[curr_node].values()), dtype=np.int)
                        cn_connected_nodes = cn_connected_nodes[cn_connected_nodes_depth == 1]
                        valid_expanded_neihbors = connected_nodes[connected_nodes_depth == len(path)]
                        valid_neighbors = np.intersect1d(valid_expanded_neihbors, cn_connected_nodes)
                        if len(valid_neighbors) != 0:
                            if by_prob:
                                numerator = self.degree[valid_neighbors + 1]
                                denominator = sum(numerator)
                                if denominator != 0:
                                    p = numerator * 1.0 / denominator
                                else:
                                    None
                            # print(len(path), node_id, curr_node, self.degree[curr_node], valid_neighbors, valid_neighbors.shape)
                            curr_node = np.random.choice(valid_neighbors, p=p)
                            path.insert(0, curr_node)
                        else:
                            break
                    arr = np.array(path) + 1
                    npad = (self.diameter + 1) - len(arr)
                    if npad > 0:
                        arr = np.lib.pad(arr, (0, npad), 'constant', constant_values=(0))
                    walks = np.append(walks, arr.reshape(1, self.diameter + 1), axis=0)
        return walks

    def walks_generator(self, data='train', by_max_degree=True, by_prob=True, shuffle=True):
        nodes = np.where(self.get_nodes(data))[0]
        if shuffle:
            indices = np.random.permutation(len(nodes))
            nodes  = nodes[indices]

        #while True:
        for node_id in nodes:
            x = self.sample_DFS_walks(node_id, by_max_degree=by_max_degree, by_prob=by_prob)
            #print ("---------------Shape: ", x.shape)
            x = np.swapaxes(x, 0, 1) # convert from (batch x step) to (step x batch)

            temp = np.array(x)>0  #get locations of all zero inputs as binary matrix
            lengths = np.sum(temp, axis=0)

            x1 = [[self.all_features[item] for item in row] for row in x] # get features for all data points
            x2 = [[self.label_cache.get(item, self.all_labels[0]) for item in row] for row in x] # get pseudo labels
            y  = [self.all_labels[node_id]]  #get tru labels for Node of interest

            #print(x, lengths)
            yield (x1, x2, y, lengths, node_id)


