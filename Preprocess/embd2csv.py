import numpy as np


def get_embeddings(data_dir, dim=128):
    f = open(data_dir, 'r')
    f.readline()#remove meta-data in first line
    data = f.read().split()
    data = [float(item) for item in data]
    embed = np.zeros((len(data)//(dim+1), dim))
    for i in range(0, len(data), dim + 1):
        embed[int(data[i])] = data[i+1 : i+1 + dim]
    return embed


np.savetxt('', get_embeddings('', dim=128), delimitter=',')
