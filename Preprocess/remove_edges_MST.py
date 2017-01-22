from __future__ import print_function

import networkx as nx
import numpy as np
import scipy
from scipy.io import loadmat, savemat


file_path = './'
dataset = 'blogcatalog'
remove_percent = [0.5, 0.4, 0.3, 0.2, 0.1]

graph = loadmat(file_path+dataset+'.mat')['network']
graph = nx.from_scipy_sparse_matrix(graph)

all_edges = graph.edges(data=False)
total_edges = len(all_edges)
print("Total_edges: ", total_edges)
#Get the edges required to keep the graph connected even after removing certain edges
MST = nx.minimum_spanning_tree(graph)
print("Done calculating MST...")

removeable_edges = list(set(all_edges) - set(MST.edges()))
np.random.shuffle(removeable_edges)

reduced_graphs = {}
for percent in remove_percent:
    if int(percent*total_edges) < len(removeable_edges):
        selected_edges = removeable_edges[int(percent*total_edges):]
    else:
        print("Not enough edges to keep the graph connected when removing: ", percent*100)
        selected_edges = removeable_edges
        exit()

    selected_edges.extend(MST.edges())
    print("Total selected edges: ", len(selected_edges))
    
    new_graph = nx.Graph()
    new_graph.add_edges_from(selected_edges)
    savemat(dataset+str(100- int(percent*100)),{'network': nx.to_scipy_sparse_matrix(new_graph, weight=None),
                                                'removed_edges':np.array(removeable_edges[:int(percent*total_edges)])})


