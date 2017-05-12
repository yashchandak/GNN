import numpy as np
from copy import deepcopy
from os import path


class DataSet(object):
    def __init__(self, cfg):
        """Construct a DataSet.
        """
        self.cfg = cfg
        self.all_labels = self.get_labels(cfg.label_path)
        self.all_features = self.get_fetaures(cfg.features_path)

        # Increment the positions by 1 and mark the 0th one as False
        self.train_nodes = np.concatenate(([False], np.load(path.join(cfg.fold_dir, 'train_ids.npy'))))
        self.val_nodes = np.concatenate(([False], np.load(path.join(cfg.fold_dir, 'val_ids.npy'))))
        self.test_nodes = np.concatenate(([False], np.load(path.join(cfg.fold_dir, 'test_ids.npy'))))
        # [!!!IMP!!]Assert no overlap between test/val/train nodes

        self.change = 0
        self.path_pred_variance = {}
        self.label_cache = np.zeros_like(self.all_labels)
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
            wce = 1 / (len(tot) * (tot * 1.0 / np.sum(tot)))
        else:
            wce = [1] * self.all_labels.shape[1]

        print("Cross-Entropy weights: ", wce)
        return wce

    def load_graph(self):
        dist = np.load(open(self.cfg.length_path, 'rb'), encoding='latin1')
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
        print(
            'Train Nodes: %d \nVal Nodes: %d \nTest Nodes: %d \nFeatures: %d \nLabels: %d \nMulti-label: %s \nDiameter: %d \nMax Degree: %d' \
            % (
            self.n_train_nodes, self.n_val_nodes, self.n_test_nodes, self.n_features, self.n_labels, self.multi_label,
            self.diameter, np.max(self.degree)))
        print('-----------------------------------------------------\n')

    def get_nodes(self, dataset):
        if dataset == 'train':
            nodes = self.train_nodes
        elif dataset == 'val':
            nodes = self.val_nodes
        elif dataset == 'test':
            nodes = self.test_nodes
        elif dataset == 'all':
            # Get all the nodes except the 0th node
            nodes = [True] * len(self.train_nodes)
            nodes[0] = False
        else:
            raise ValueError
        return nodes

    def update_label_cache(self, data, predictions, ids):
        if ids is None:
            self.label_cache[self.get_nodes(data)] = predictions
        else:
            self.label_cache[ids] = predictions

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

    def sample_DFS_walks(self, node_id, n_walks=1, by_max_degree=False, by_prob=False,):
        # dist array starts from 0 :/
        walks = np.empty((0, self.diameter + 1), int)
        connected_nodes = np.array(list(self.dist[node_id - 1].keys()), dtype=np.int)
        connected_nodes_depth = np.array(list(self.dist[node_id - 1].values()), dtype=np.int)

        if by_max_degree:
            immediate_neighbors = connected_nodes[connected_nodes_depth == 1]
            additional_neighbors_reqd = max(0, self.cfg.max_walks - self.degree[node_id])
            p = None
            if immediate_neighbors.shape[0] == 0:
                path = [node_id-1]
                arr = np.array(path) + 1
                npad = (self.diameter + 1) - len(arr)
                if npad > 0:
                    arr = np.lib.pad(arr, (0, npad), 'constant', constant_values=(0))
                walks = np.append(walks, arr.reshape(1, self.diameter + 1), axis=0)
            else:
                if additional_neighbors_reqd != 0:
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
            nodes = nodes[indices]

        # while True:
        for node_id in nodes:
            x = self.sample_DFS_walks(node_id, by_max_degree=by_max_degree, by_prob=by_prob)
            # print ("---------------Shape: ", x.shape)
            x = np.swapaxes(x, 0, 1)  # convert from (batch x step) to (step x batch)

            temp = np.array(x) > 0  # get locations of all zero inputs as binary matrix
            lengths = np.sum(temp, axis=0)

            x1 = [[self.all_features[item] for item in row] for row in x]  # get features for all data points
            x2 = [[self.label_cache[item] for item in row] for row in x]  # get pseudo labels
            y = [self.all_labels[node_id]]  # get tru labels for Node of interest

            yield (x, x1, x2, lengths, y, node_id)
