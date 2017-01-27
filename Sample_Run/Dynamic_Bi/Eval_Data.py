from __future__ import print_function
import numpy as np
from scipy.io import loadmat

class Data():

    def __init__(self, cfg):
        self.cfg = cfg
        self.labels     = self.get_labels(self.cfg.label_file)
        self.embeddings = self.get_embeddings_csv(self.cfg.embed_file)
        self.set_training_validation(10,1) #Some value just for initialization
        self.has_more   = True
        self.index      = 0


    def set_training_validation(self,percent,fold):
        
        training   = np.load(self.cfg.directory +'labels/'+str(percent)+'/'+str(fold)+'/train_ids.npy' )#self.splits[train]
        validation = np.load(self.cfg.directory +'labels/'+str(percent)+'/'+str(fold)+'/val_ids.npy' )#self.splits[valid]

        #Get the positions where boolean values are True
        #concatenate training and validation used for language model
        self.training = np.concatenate((np.where(training)[0], np.where(validation)[0]))
        self.test     = np.where(np.load(self.cfg.directory +'labels/'+str(percent)+'/'+str(fold)+'/test_ids.npy' ))[0]


    def get_labels(self, path):
        labels = loadmat(path)['labels']
        return labels
    
    def get_embeddings(self, data_dir):
        #Read from key-value format of Gensim dumps
        f = open(data_dir, 'r')
        f.readline()#remove meta-data in first line
        data = f.read().split()
        data = [float(item) for item in data]
        embed = {}
        for i in range(0, len(data), self.cfg.input_len + 1):
            embed[int(data[i])] = data[i+1 : i+1 + self.cfg.input_len]
            #print(data[i], data[i+1:self.cfg.input_len + 1])
        return embed

    def get_embeddings_csv(self, data_dir):
        embed = np.loadtxt(data_dir, dtype='float', delimiter=',')
        return embed
    
    def read_data(self, data_dir):
        nodes = []
        data = open(data_dir).read().split()
        return [int(item) for item in data]

    def reset(self):
        self.index = 0
        self.has_more = True

    def next_batch(self):
        #Batch-wise feeding for Neural net based classifier
        inp_nodes  = np.zeros((self.cfg.batch_size, self.cfg.input_len))
        inp_labels = np.zeros((self.cfg.batch_size, self.cfg.label_len))
        if self.has_more:
            for i, val in enumerate(self.training[self.index: self.index+self.cfg.batch_size]):              
                inp_nodes[i] = self.embeddings[int(val)]
                inp_labels[i] = self.labels[val]

            self.index += self.cfg.batch_size

        #Check if next batch is possible
        if self.index+self.cfg.batch_size >= len(self.training):
            self.has_more = False

        return inp_nodes, inp_labels


    def get_all_nodes_labels(self, data):
        #Get the embedding and corresponding labels at once for all nodes
        inp_nodes  = np.zeros((len(data), self.cfg.input_len))
        inp_labels = np.zeros((len(data), self.cfg.label_len))
        for i,val in enumerate(data):
            inp_nodes[i] = self.embeddings[val]
            inp_labels[i] = self.labels[val]

        return inp_nodes, inp_labels
