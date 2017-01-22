from __future__ import print_function

import networkx as nx
import numpy as np
import scipy
from scipy.io import loadmat, savemat



file_path = './'
dataset = 'blogcatalog90'

mat = loadmat(file_path+dataset+'.mat')
graph = mat['network']
graph = nx.from_scipy_sparse_matrix(graph)
selected_edges = graph.edges(data=False)

dataset_original = 'blogcatalog'
mat_o = loadmat(file_path+dataset_original+'.mat')
graph_o = mat_o['network']
graph_o = nx.from_scipy_sparse_matrix(graph_o)
edges_all = graph_o.edges(data=False)

removed_edges = list(set(edges_all) - set(selected_edges))
mat['removed_edges'] = np.array(removed_edges)

savemat(file_path+dataset, mat)
