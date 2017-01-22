from __future__ import print_function
from scipy.io import savemat, loadmat
import numpy as np
import networkx as nx

f = open('karate.edgelist', 'r').read().split()
f = [int(item)-1 for item in f]
edges = [(f[idx], f[idx+1]) for idx in range(0, len(f), 2)]

G = nx.Graph()
G.add_edges_from(edges)
savemat('karate.mat', {'network':nx.to_scipy_sparse_matrix(G, weight=None)})


