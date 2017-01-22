from __future__ import print_function
import re

path = '/home/priyesh/Desktop/datasets/youtube/'
f = open(path + 'edges.csv', 'r').read()
data = re.split(',|\n', f)


f = open(path + 'directed_weighted_edges.txt', 'w')
for i in range(0,len(data),2):
    f.write(data[i] + ' ' + data[i+1] + ' 1\n')
    f.write(data[i+1] + ' ' + data[i] + ' 1\n')
 
f.close()
