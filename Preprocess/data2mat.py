from __future__ import print_function

import scipy.io as sio
from scipy.sparse import dok_matrix 
import numpy as np

def get_ID(node2ID, node):
    if node2ID.get(node, -1) != -1:
        return node2ID[node]

    else:
        node2ID[node] = len(node2ID.keys())
        return node2ID[node]

    
def read_graph(path):
    #creates sparse graph
    
    f = open(path, 'r')
    vals = f.read().split()
    f.close()

    node2ID = {}
    graph   = {}
    dct     = {}   

    for idx in range(0, len(vals), 2):
        node_to   = get_ID(node2ID, vals[idx])
        node_from = get_ID(node2ID, vals[idx+1])

        dct[(node_from, node_to)] = 1

        neighbors = graph.get(node_from, [])
        neighbors.append(node_to)
        graph[node_from] = neighbors

    n = len(node2ID.keys() )
    smat = dok_matrix((n, n))
    smat.update(dct)
    #print(sorted(node2ID.keys()), len(set(node2ID.values())))
    return smat, node2ID, graph
        
    
def read_data(path, node2ID):

    f = open(path, 'r')
    vals = f.read().split()
    f.close()

    all_labels= {}
    ID_labels = {}
    ID_feats  = {}

    #format: <paper_id> <word_attributes> <class_label>
    for idx in range(0,len(vals), feature_len + 2):

        #Get the ID of the cuurent node
        ID    = node2ID[vals[idx]]

        #insert the label mapping into the dictionary if not present already
        l = vals[idx + feature_len + 1]
        all_labels[l] = all_labels.get(l, len(all_labels.keys()))

        #get label for cuurent node
        label = all_labels[l]

        #Create the attribute/feature vector
        val  = vals[idx+1: idx+1+feature_len]
        val  = [int(item) for item in val]
        feats = []
        for i in range(feature_len):
            if val[i] == 1:
                feats.append(i)

        ID_labels[ID] = label 
        ID_feats[ID]  = feats    

    #print(ID_feats)
    return ID_feats, ID_labels


path   = '/home/Desktop/datasets/'
dataset= 'citeseer'
folder = dataset+'/'
data   = dataset+'.content'
edges  = dataset+'.cites'

"""
#Citeseer
node_count  = 3327 #3312
edge_count  = 4732
feature_len = 3703
label_len   = 6

"""
"""
#cora
node_count  = 2708
edge_count  = 5429
feature_len = 1433
label_len   = 7

"""
#webKB
#node_count  = 2708
#edge_count  = 5429
feature_len = 1703
label_len   = 5

if __name__ == '__main__':

    smat, node2ID, graph   = read_graph(path+folder+edges)
    print("Graph ready...")
    
    features, labels = read_data(path+folder+data, node2ID)
    print("features and labels ready...")

    mat  ={'network':smat, 'labels':labels, 'features':features}
    sio.savemat(path+folder+dataset+'.mat', mat)
    print("Data saved to: ", path+folder+dataset+'.mat')


    #Do sanity check
    print (len(node2ID.keys()),  len(set(node2ID.values())), len(set(labels.values())), sum([len(v) for k,v in graph.items()]))
    #assert len(node2ID.keys()) == node_count
    #assert len(set(node2ID.values())) == node_count
    assert len(set(labels.values())) == label_len
    #assert sum([len(v) for k,v in graph.items()]) == edge_count
