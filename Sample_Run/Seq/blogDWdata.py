from __future__ import generators
#import arff
import collections
import numpy as np
from tensorflow.python.framework import dtypes
from random import shuffle
from scipy.io import loadmat

class DataSet(object):

  def __init__(self,cfg):
    """Construct a DataSet.
    """
    self.all_walks    = np.load(cfg.walks_dir) # {node : [walks * steps]} 
    self.all_labels   = self.get_labels(cfg.label_dir)
    self.all_features = self.get_fetaures(cfg.features_dir) 
    self.train_nodes  = np.load(cfg.label_fold_dir+'train_ids.npy')
    self.val_nodes    = np.load(cfg.label_fold_dir+'val_ids.npy')
    self.test_nodes   = np.load(cfg.label_fold_dir+'test_ids.npy')
    #Assert no overlap
    
  def next_batch(self,dataset, batch_size=None,shuffle=False):

    if dataset == 'train':
      nodes = self.train_nodes
    elif dataset == 'val':
      nodes = self.val_nodes
    elif dataset == 'test':
      nodes = self.test_nodes
    
    label_len = np.shape(self.all_labels)[1]

    data, node_seq =[],[]
    for idx, val in enumerate(nodes):
      node = idx + 1
      if val:
        data.extend(self.all_walks[node])  #stack all the walks
        node_seq.extend([node]*len(walks)) #stack their corresponding end nodes

    if shuffle:
        indices = np.random.permutation(len(data))
        data    = data[indices]
        node_seq= node_seq[indices]

    for i in range(0,len(data), batch_size):
        x = data[i: i+batch_size]
        x = [[features[item] for item in row] for row in x]#(batch x step x feature_size)
        y = [[self.labels.get(item, self.labels[0]) for item in row] for row in x]#(batch x step x label_size)
        x_labels = [label[seq[-1] for seq in x] #label for last node in sequence (batch x label_size)
        
        #get features for all data points
        #convert from (batch x step x feature_size) to (step x batch x feature_size)
        x = [[features[item] for item in row] for row in x]
        x = np.swapaxes(x, 0,1)
                    
        yield (x, np.array(x_labels))


  def get_fetaures(self, path):
    all_features      = np.load(path)
    all_features      = all_features.astype(np.float32, copy=False) #Required conversion for Python3
    #Serves 2 purpose:
    #a) add feature for dummy node 0 a.k.a <EOS> and <unlabeled>
    #b) increments index of all features by 1, thus aligning it with indices in walks
    all_features =  np.concatenate(([np.zeros(all_features.shape[1])], all_features), 0)
    return all_features
  
  def get_labels(self, path):
    all_labels = loadmat(path)['labels']
    #Labels start with node '0'; Walks_data with node '1'
    #To get corresponding mapping, increment the label node number by 1
    #add label for dummy node 0 a.k.a <EOS> and <unlabeled>
    #Mark the first position as '1' to indicate dummy label
    z = np.zeros(all_labels.shape[1])
    z[0] =1
    all_labels = np.concatenate(([z], all_labels), 0)
    
    return all_labels
      
 
