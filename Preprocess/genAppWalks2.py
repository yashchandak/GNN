import networkx as nx
import numpy as np
from random import randint
from scipy.io import loadmat

dataset = 'Homo_sapiens'
x = loadmat(dataset+'.mat')
x = x['network']
G = nx.from_scipy_sparse_matrix(x)
del x


max_walks_per_node = 640
length = nx.all_pairs_shortest_path_length(G)
np.save(dataset+'-length.npy', length)
#length = np.load('length.npy').item()
DEPTH = np.max([dist for node, neigh in length.items() for n, dist in neigh.items()]) #5
nodes = np.array(range(len(length)))


f = open(dataset+'_walks.csv','w')
f.close()

for node in nodes:
    neighborhood = np.array(length[node].values())
    f = open(dataset+'_walks'+str(max_walks_per_node)+'.csv','a')
    walks = np.empty((0,DEPTH+1), int)
    while len(walks) <= max_walks_per_node:
        path = []
        curr_node = node
        while len(path) <= DEPTH:
            path.append(curr_node)
            curr_neighborhood = np.array(length[curr_node].values())
            curr_neighbors = nodes[curr_neighborhood==1]
            valid_neighbors = nodes[neighborhood==len(path)]
            valid_candidates = np.intersect1d(valid_neighbors,curr_neighbors)
            if len(valid_candidates) != 0:
                curr_node = valid_candidates[randint(0,len(valid_candidates)-1)]
            else:
                break
        arr = np.array(path) + 1
        npad = (DEPTH+1)-len(arr)
        if npad > 0:
          arr = np.lib.pad(arr,(0,npad),'constant',constant_values=(0))
	#reverse the walks
        #arr = arr[::-1]
        arr.reshape(1,DEPTH+1)
        walks = np.append(walks,arr.reshape(1,DEPTH+1),axis=0)
    np.savetxt(f,walks,delimiter=' ',fmt='%i')
    if node % 500 == 0:
        print(node)
f.close()

