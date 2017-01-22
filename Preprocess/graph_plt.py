import networkx as nx
from scipy.io import loadmat
import matplotlib.pyplot as plt

dataset = 'citeseer'
x = loadmat(dataset+'.mat')
x = x['network']
G = nx.from_scipy_sparse_matrix(x)
pos = nx.spring_layout(G)

plt.figure(num=None, figsize=(20, 20), dpi=80)
plt.axis('off')
fig = plt.figure(1)

#nx.draw_networkx_nodes(G,pos,node_size=weight)
nx.draw_networkx_nodes(G,pos)
nx.draw_networkx_edges(G,pos)


cut = 1.00
xmax = cut * max(xx for xx, yy in pos.values())
ymax = cut * max(yy for xx, yy in pos.values())
plt.xlim(0, xmax)
plt.ylim(0, ymax)

#plt.draw()
plt.savefig('a.png')
plt.gcf().clear()
