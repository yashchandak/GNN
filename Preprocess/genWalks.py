import networkx as nx
import numpy as np
from scipy.io import loadmat

x = loadmat('blogcatalog.mat')
x = x['network']
G = nx.from_scipy_sparse_matrix(x)
del x

DEPTH = 10
#length = nx.all_pairs_shortest_path_length(G)
length = np.load('length.npy').item()
nodes = np.array(range(1,len(length)+1))

for node in nodes:
  neighborhood = np.array(length[node].values())
  for d in range(1,DEPTH+1):
    neighbors = nodes[neighborhood==d]
    for neighbor in neighbors:
      paths = nx.all_shortest_paths(G,source=node,target=neighbor)
      f = open('KD_walks.csv','a')
      walks = np.empty((0,DEPTH+1), int)
      for path in paths:
        arr = np.array(path)
        npad = (DEPTH+1)-len(arr)
        if npad > 0:
          arr = np.lib.pad(arr,(0,npad),'constant',constant_values=(0))
        arr = arr[::-1]
	arr.reshape(1,DEPTH+1)
        walks = np.append(walks,arr.reshape(1,DEPTH+1),axis=0)

      np.savetxt(f,walks,delimiter=',',fmt='%i')
      if node % 500 == 0:
	print(node)

