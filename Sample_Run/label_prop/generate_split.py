import numpy as np
from random import shuffle

root_path = '/home/priyesh/Desktop/Codes/score/'
project_name = 'MLP/'
dataset = 'BlogDWdata/'

node_count = 10312

data = {}
#Training, validation, testing
splits = [ [0.1, 0.9, 0.0], [0.5, 0.5, 0.0], [0.9, 0.1, 0.0]]
#assert np.sum(split) == 1.0

for i in range(10):
    for split in splits:
	nodes = np.arange(node_count)
	shuffle(nodes)
	node_train = nodes[0: int(split[0]*node_count)]
	node_valid = nodes[int(split[0]*node_count): int((split[0]+split[1])*node_count)]
	node_test  = nodes[int((split[0]+split[1])*node_count):]
	
        data[('train',i, int(split[0]*100))] = node_train
	data[('valid',i, int(split[0]*100))] = node_valid
	data[('test',i, int(split[0]*100))] = node_test	

np.save(root_path + project_name + dataset + 'splits.dict', data)



