import numpy as np

folder = '/home/priyesh/Desktop/datasets/blogcatalog/blogcatalog60/'
data = 'blogcatalog60'
fname = folder+'blogcatalog60_walks80.csv'

with open(fname) as f:
	content = f.read().splitlines()

indices = np.random.permutation(len(content))
chardata = np.asarray(content)
chardata = chardata[indices]
n_val = int(len(content)*0.1)
val_x = chardata[:n_val]
train_x = chardata[(n_val+1):]

np.savetxt(folder+'train_'+data+'.txt', train_x, delimiter=" ",fmt='%s')
np.savetxt(folder+'val_'+data+'.txt', val_x, delimiter=" ",fmt='%s')
np.savetxt(folder+'p_'+data+'.txt',chardata, delimiter=" ",fmt='%s')
