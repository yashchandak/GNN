from __future__ import print_function
import numpy as np

import networkx as nx
from collections import OrderedDict
from scipy.io import loadmat, savemat
import os

"""
IMP: Nodes start from 0
"""

dataset = 'facebook'
source_path = '../Sample_Run/Datasets/%s/DCI_format/'%(dataset)
dest_path = '../Sample_Run/Datasets/%s/ours/'%(dataset)


def net2mat():
    f = open(source_path+dataset+'.net', 'rb')
    f.readline() #remove first line
    G = nx.parse_edgelist(f, delimiter=',',nodetype=int, data=(('weight',float),))

    G_sparse = nx.to_scipy_sparse_matrix(G)
    savemat(dest_path+'adjmat', {'adjmat':G_sparse})


def dat2feats_labels():
    f = open(source_path + dataset + '.dat', 'rb')
    print(f.readline())  # remove first line
    feats = {}
    labels = {}
    for line in f:
        node, feat, label =  line.strip().split(',')
        feats[int(node)] = list(feat)
        labels[int(node)] = list(label)

    #dict.items are arranged in ascending order by the key value
    feats  = OrderedDict(sorted(feats.items(), key=lambda kv: kv[0]))
    labels = OrderedDict(sorted(labels.items(), key=lambda kv: kv[0]))
    feat_list = np.array([v for k, v in feats.items()], dtype=int)
    label_list= np.array([v for k, v in labels.items()], dtype=int)

    print(np.shape(feat_list), np.shape(label_list))

    np.save(dest_path+'features.npy', feat_list)
    np.save(dest_path+'labels.npy', label_list)


def create_graph_mapping():
    #create mappings from labels file
    f = open(source_path+dataset+'.edges', 'rb')
    G = nx.Graph()
    for line in f:
        x, y = line.strip().split('::')
        G.add_edge(x,y)

    remove = []
    map = {}
    ids = []
    ctr = 0
    f2 = open(source_path+dataset+'.attr', 'rb')
    for line in f2:
        l = line.strip().split('::')
        node = l[0]
        ids.append(node)
        map[node] = ctr
        ctr += 1

    print('Total nodes: %d, Singleton nodes removed: %d'%(len(ids),len(remove)))
    np.save(dest_path+'ids', ids)
    np.save(dest_path+'map', map)
    print('Done creating Mapping for %d node ids'%len(ids))

    G_mapped = nx.Graph()
    #add edges
    for u,v in G.edges():
        u,v = map[u], map[v]
        G_mapped.add_edge(u,v)

    #add singleton nodes
    G_mapped.add_nodes_from(set(map)-set(G.nodes()))

    G_sparse = nx.to_scipy_sparse_matrix(G_mapped)
    savemat(dest_path+'adjmat', {'adjmat':G_sparse})
    print('Done creating Mapped Graph')


def attr2feats():
    map = np.load(dest_path+'map.npy').item()
    f = open(source_path+dataset+'.attr', 'rb')
    feats = {}
    unmapped = []
    for line in f:
        l = line.split('::')
        pos = map.get(l[0], -1)
        if pos != -1:
            feats[pos] = l[1:]
        else:
            unmapped.append(l[0])

    print('%d Nodes dont have a mapping!'%(len(unmapped)))

    feats = OrderedDict(sorted(feats.items(), key=lambda kv: kv[0]))
    feat_list = np.array([v for k, v in feats.items()], dtype=int)

    np.save(dest_path+'features.npy', feat_list)
    print('Done creating', np.shape(feat_list), 'features')


def lab2labels(max_len =2):
    #Supports multi-labels per node also
    map = np.load(dest_path+'map.npy').item()
    f = open(source_path+dataset+'.lab', 'rb')
    labels = {}
    unmapped = []
    for line in f:
        l = line.split('::')
        pos = map.get(l[0], -1)
        if pos != -1:
            temp = labels.get(pos, [0]*max_len)
            temp[int(l[1])] = 1
            labels[pos] = temp
        else:
            unmapped.append(l[0])

    print('%d Nodes dont have a mapping!'%(len(unmapped)))
    labels = OrderedDict(sorted(labels.items(), key=lambda kv: kv[0]))
    label_list = np.array([v for k, v in labels.items()], dtype=int)

    np.save(dest_path+'labels.npy', label_list)
    print('Done creating', np.shape(label_list), 'labels')

def createfolds(trials =10, folds=17):

    map = np.load(dest_path+'map.npy').item()
    size = len(map.values())

    for trial in range(trials):
        #They have a common validation set for each trial
        val_file = source_path+dataset+'_trial_%d_val.txt'%(trial)
        val = np.zeros(size, dtype=bool)
        val[[map[node] for node in np.loadtxt(val_file, dtype=str)]] = True

        for fold in range(folds):
            train_file = source_path+dataset+'_trial_%d_fold_%d.txt'%(trial, fold)
            train = np.zeros(size, dtype=bool)
            train[[map[node] for node in np.loadtxt(train_file, dtype=str)]] = True

            #Add train + Validation sets and then invert it
            test = -(train+val)

            path = dest_path + 'labels/%d/%d/'% (trial, fold)
            if not os.path.exists(path):
                os.makedirs(path)

            np.save(path + 'val_ids', val)
            np.save(path + 'train_ids', train)
            np.save(path + 'test_ids', test)

    print('Done creating Test, Valid, Train samples')
    return


create_graph_mapping()
attr2feats()
lab2labels()
createfolds()