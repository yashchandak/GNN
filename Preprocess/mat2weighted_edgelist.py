from __future__ import print_function
from scipy.io import loadmat
import networkx as nx

path = '/home/priyesh/Desktop/datasets/blogcatalog/blogcatalog90/'
dataset = 'blogcatalog90'
mat = loadmat(path+dataset+'.mat')
mat = mat['network']
G = nx.from_scipy_sparse_matrix(mat)

f = open(path + dataset + '_undirected_unweighted_edges.txt', 'w')
for n1, n2, w in list(G.edges_iter(data='weight', default=1)):
    f.write(str(n1) + ' ' + str(n2) + '\n')# ' ' + str(int(w)) + '\n')
    #f.write(str(n2) + ' ' + str(n1) + ' ' + str(int(w)) + '\n')
 
f.close()
