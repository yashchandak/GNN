from __future__ import print_function

import numpy as np
from scipy.io import loadmat, savemat
from scipy.spatial.distance import pdist, squareform
import networkx as nx

def get_topk_edges(distances, nodes, k=5):
    assert k < nodes**2
    
    topk = np.argpartition(distances, k)[:k]
    topk_edges = []
    for pos in topk:
        #print(distances[pos])
        n1, n2 = pos//nodes, pos%nodes
        topk_edges.append((n1,n2))

    print(topk_edges)
    return topk_edges
    
    
def read_embeddings(data_dir, csv, dim=128):
    if csv:
        return np.loadtxt(data_dir, delimiter = ',')

    f = open(data_dir, 'r')
    f.readline()#remove meta-data in first line
    data = f.read().split()
    data = [float(item) for item in data]
    embed = np.zeros((len(data)//(dim+1), dim))
    for i in range(0, len(data), dim + 1):
        embed[int(data[i])] = data[i+1 : i+1 + dim]
    return embed

def p_at_k(topk_edges, removed_edges):
    k = len(topk_edges)
    correct_edges = len(set(topk_edges) & set(removed_edges))
    print("wrong: ", set(topk_edges) - set(removed_edges))
    return correct_edges/k
    


""" ====== IMP: Only for symmetric graphs without self loops ======"""
# mat_dir = 'blogcatalog90.mat'
mat_dir = 'karate.mat'
# embd_dir = 'blogcatalog90_DW.embd'
embd_dir = 'Karate_data.embd'
csv = True
ks = [1, 3 , 5, 10, 15, 20]#, 1000, 5000, 10000, 50000]

embd = read_embeddings(embd_dir, csv=csv, dim=128)
print("Read embeddings from: ", embd_dir)

mat = loadmat(mat_dir)
graph = nx.from_scipy_sparse_matrix(mat['network'])
selected_edges = graph.edges(data=False)

np.random.shuffle(selected_edges)
removed_edges  = selected_edges[:20]#mat['removed_edges'] 
selected_edges  = selected_edges[20:]#mat['removed_edges']
print(removed_edges, '\n', selected_edges)

print("Read graph from: ", mat_dir)
del graph, mat

distances = squareform(pdist(embd,'euclidean'))
print("Calculated pair-wise distances...")

nodes = embd.shape[0]
precisions = {}

#set the distnces among nodes of existing edges 
for n1, n2 in selected_edges:
    distances[n1][n2] = distances[n2][n1] = 999

#Make lower triangle and diagnol -inf
for n1 in range(nodes):
    for n2 in range(0, n1+1):
        distances[n1][n2] = np.inf
print("Done removing existing edges and making matrix as upper triangle...\n", distances)

#Assert n2>n1 for removed edges and Convert to set
for n1, n2 in removed_edges:
    assert n2>n1 , str(n2)+str(n1)
removed_edges = [(n1, n2) for n1, n2 in removed_edges]
#print(removed_edges)


distances = np.ndarray.flatten(distances)
for k in ks:
    topk_edges = get_topk_edges(distances, nodes, k)
    precisions[k] = p_at_k(topk_edges, removed_edges)
    print("P@", k, ": ", precisions[k])
