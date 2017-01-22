import networkx as nx
from scipy.io import loadmat
x = loadmat(dataset)
dataset='blogcatalog.mat'
x = loadmat(dataset)
x = x['network']
G = nx.from_scipy_sparse_matrix(x)
del x
f=open("BC_DW.edgelist",'wb')
nx.write_edgelist(G, f)


