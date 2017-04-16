from __future__ import print_function
import numpy as np
from scipy.io import loadmat

class Data():

    def __init__(self, cfg):
        self.cfg = cfg
        self.labels     = np.load(self.cfg.label_dir)
        self.embeddings = np.load(self.cfg.embed_file).astype(np.float32)
        self.set_training_validation(cfg.training_percents[0],cfg.num_shuffles[0]) #First value just for initialization


    def set_training_validation(self,percent,fold):
        
        self.training   = np.where(np.load(self.cfg.directory +'labels/'+str(percent)+'/'+str(fold)+'/train_ids.npy' ))[0]
        self.validation = np.where(np.load(self.cfg.directory +'labels/'+str(percent)+'/'+str(fold)+'/val_ids.npy' ))[0]
        self.test       = np.where(np.load(self.cfg.directory +'labels/'+str(percent)+'/'+str(fold)+'/test_ids.npy' ))[0]
        print(len(self.training), len(self.validation), len(self.test))

    def next_batch(self, data):
        #Batch-wise feeding for Neural net based classifier
        pos = []
        if data == 'train':
            pos = self.training
        elif data == 'val':
            pos = self.validation
        elif data == 'test':
            pos = self.test
        else:
            raise ValueError

        batch_size = self.cfg.batch_size
        tot = len(pos)//batch_size
        for i in range(0, len(pos), batch_size):
            x = self.embeddings[pos[i: i + batch_size]]
            y = self.labels[pos[i: i + batch_size]]
            yield (x, y, tot)


    def get_all_nodes_labels(self, data):

        pos = []
        if data == 'train_val':
            #Lib linear takes training and validation sets together
            pos = self.training
            pos  = np.concatenate((pos,self.validation))

        elif data == 'train':
            pos = self.training

        elif data == 'test':
            pos = self.test
        else:
            raise ValueError

        x = self.embeddings[pos[:]]
        y = self.labels[pos[:]]
        return x, y
