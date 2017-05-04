from __future__ import  print_function
import numpy as np
from os import path
from copy import deepcopy


data_dir = 'citeseer'
walk_path = path.join(data_dir, 'walks')
val_files = ['val_walks_80.txt']


class Dataset(object):

    def __init__(self, data_dir, max_diameter=10):
        #Node Ids start from 1
        self.data_dir = data_dir
        self.true_labels = self.load_labels()
        self.features = self.load_features()
        self.length, self.diameter, self.degree = self.load_graph()
        self.diameter = min(max_diameter, self.diameter)

        self.n_nodes, self.n_labels = self.true_labels.shape
        self.n_nodes -= 1
        _, self.n_features = self.features.shape
        self.multi_label = self.is_multilabel()
        self.sampler_reset = False
        self.print_statistics()

    def load_labels(self):
        labels = np.load(path.join(self.data_dir, 'labels.npy')).astype(np.int32, copy=False)
        labels = np.concatenate((np.zeros((1, labels.shape[1])), labels), axis=0)
        return labels

    def load_features(self):
        features = np.load(path.join(self.data_dir, 'features.npy')).astype(np.float32, copy=False)
        features = np.concatenate((np.zeros((1, features.shape[1])), features), axis=0)
        return features

    def load_graph(self):
        length = np.load(open(self.data_dir+'/length.pkl', 'rb'), encoding='latin1')
        diameter = np.max([dist for node, neigh in length.items() for n, dist in neigh.items()])
        degree = [np.sum(np.array(list(length[node_id].values()), dtype=np.int) == 1) for node_id in range(len(length))]
        degree.insert(0, 0)
        return length, diameter, np.array(degree)

    def get_degree(self, node_id):
        return self.degree[node_id]

    def is_multilabel(self):
        sum = np.count_nonzero(self.true_labels)
        multi_label = False
        if sum > self.n_nodes:
            multi_label = True
        return multi_label

    def print_statistics(self):
        print('############### DATASET STATISTICS ####################')
        print('Nodes: %d \nFeatures: %d \nLabels: %d \nMulti-label: %s \nDiameter: %d \nMax Degree: %d'\
              %(self.n_nodes, self.n_features, self.n_labels, self.multi_label, self.diameter, np.max(self.degree)))
        print('############### DATASET STATISTICS ####################')


    def get_dataset_sequences(self, file):
        for line in open(path.join(self.data_dir, file)).readlines():
            yield line

    def next_sequence_loop(self, sequence_generator):
        gen = sequence_generator.__iter__()
        while True:
            for lines in gen:
                seq = []
                for line in lines:
                    seq.append([int(node_id) for node_id in line])
                yield np.asarray(seq, dtype=np.int32)
                #yield np.asarray(seq, dtype=np.int32).squeeze() #Don't squeeze that will help us count the n_walks
            gen = sequence_generator.__iter__()


    def get_batches(self, batch_size, total_batches, generator):
        batch = []
        batch_cnt = 0
        n_batches = 0
        exit_flag = 0
        for seq in self.next_sequence_loop(generator):
            batch.append(seq)
            batch_cnt += 1
            if batch_cnt < batch_size:
                continue
            else:
                res = deepcopy(batch)
                batch = []
                batch_cnt = 0
                n_batches += 1
                yield res

            if n_batches >= total_batches:
                break


    def sample_DFS_walks(self, node_id, n_walks=1, by_max_degree=False, by_prob=False, max_walks=5):
        #Length array starts from 0 :/
        walks = np.empty((0, self.diameter + 1), int)
        connected_nodes = np.array(list(self.length[node_id - 1].keys()), dtype=np.int)
        connected_nodes_depth = np.array(list(self.length[node_id - 1].values()), dtype=np.int)

        if not by_max_degree:
            p = None
            while len(walks) < n_walks:
                path = list()
                path.append(node_id-1)
                valid_neighbors = connected_nodes[connected_nodes_depth == 1]
                if by_prob:
                    denominator = sum(self.degree[valid_neighbors+1])
                    if denominator != 0:
                        print (p, denominator)
                        p = self.degree[valid_neighbors+1]*1.0/ denominator #multiply by 1.0 for python2
                    else:
                        p = None
                curr_node = np.random.choice(valid_neighbors, p=p)
                path.append(curr_node)
                while len(path) <= self.diameter:
                    cn_connected_nodes = np.array(list(self.length[curr_node].keys()), dtype=np.int)
                    cn_connected_nodes_depth = np.array(list(self.length[curr_node].values()), dtype=np.int)
                    cn_connected_nodes = cn_connected_nodes[cn_connected_nodes_depth == 1]
                    valid_expanded_neihbors = connected_nodes[connected_nodes_depth == len(path)]
                    valid_neighbors = np.intersect1d(valid_expanded_neihbors, cn_connected_nodes)
                    if len(valid_neighbors) != 0:
                        if by_prob:
                            numerator = self.degree[valid_neighbors+1]
                            denominator = sum(numerator)
                            if denominator != 0:
                                p = numerator / denominator
                            else:
                                None
                        curr_node = np.random.choice(valid_neighbors, p=p)
                        path.append(curr_node)
                    else:
                        break
                arr = np.array(path) + 1 #Node_ids start from 1 in path
                npad = (self.diameter + 1) - len(arr)
                if npad > 0:
                    arr = np.lib.pad(arr, (0, npad), 'constant', constant_values=(0))
                arr.reshape(1, self.diameter + 1)
                walks = np.append(walks, arr.reshape(1, self.diameter + 1), axis=0)
        else:
            immediate_neighbors = connected_nodes[connected_nodes_depth == 1]
            additional_neighbors_reqd = max(0, max_walks - self.degree[node_id])
            immediate_neighbors = np.append(immediate_neighbors, np.random.choice(immediate_neighbors, size=additional_neighbors_reqd, replace=True))
            p = None
            for neighbor_id in immediate_neighbors:
                curr_node = neighbor_id
                path = [curr_node, node_id - 1]

                while len(path) <= self.diameter:
                    cn_connected_nodes = np.array(list(self.length[curr_node].keys()), dtype=np.int)
                    cn_connected_nodes_depth = np.array(list(self.length[curr_node].values()), dtype=np.int)
                    cn_connected_nodes = cn_connected_nodes[cn_connected_nodes_depth == 1]
                    valid_expanded_neihbors = connected_nodes[connected_nodes_depth == len(path)]
                    valid_neighbors = np.intersect1d(valid_expanded_neihbors, cn_connected_nodes)
                    if len(valid_neighbors) != 0:
                        if by_prob:
                            numerator = self.degree[valid_neighbors+1]
                            denominator = sum(numerator)
                            if denominator != 0:
                                p = numerator*1.0 / denominator
                            else:
                                None
                        #print(len(path), node_id, curr_node, self.degree[curr_node], valid_neighbors, valid_neighbors.shape)
                        curr_node = np.random.choice(valid_neighbors, p=p)
                        path.insert(0,curr_node)
                    else:
                        break
                arr = np.array(path) + 1
                npad = (self.diameter + 1) - len(arr)
                if npad > 0:
                    arr = np.lib.pad(arr, (0, npad), 'constant', constant_values=(0))
                walks = np.append(walks, arr.reshape(1, self.diameter + 1), axis=0)
        return walks

    def mixed_walks_generator(self, n_walks=1, by_max_degree=False, by_prob=False):
        node_id = 0
        while True:
            if self.sampler_reset is True:
                node_id = 0
            node_id += 1
            #print(node_id)
            #yield self.sample_DFS_walks(node_id, n_walks=n_walks, by_max_degree=by_max_degree, by_prob=by_prob).__next__()
            yield self.sample_DFS_walks(node_id, n_walks=n_walks, by_max_degree=by_max_degree, by_prob=by_prob)


# ds = Dataset(data_dir)
# cnt = 0
#
# node_id=1
#walks = ds.sample_DFS_walks(node_id=1, n_walks=3, by_max_degree=False, by_prob=False)
#walks = ds.sample_DFS_walks(node_id=1, n_walks=1, by_max_degree=False, by_prob=True)
#walks = ds.sample_DFS_walks(node_id=node_id, n_walks=1, by_max_degree=True, by_prob=True)
#print(ds.degree[node_id], walks.shape)
#print(walks[1:5])

# batch_size = 1
# total_batches = 3
#
# for batch_id, data in enumerate(ds.get_batches(batch_size, total_batches, ds.mixed_walks_generator(n_walks=1, by_max_degree=False, by_prob=False))):
#     tmp = []
#     for sample_id in range(batch_size):
#         tmp.append(data[sample_id].shape)
#     print([batch_id, len(data), tmp])
#

# for _, data in enumerate(ds.get_batches(batch_size, total_batches, ds.get_dataset_sequences('walks/test_walks.txt'))):
#     #      print([len(data), data[0].shape])
#     #print(data[0])
#     continue

