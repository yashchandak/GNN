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
    self.all_walks    = np.fliplr(np.loadtxt(cfg.walks_dir)) #reverse the sequence
    self.node_seq     = self.all_walks[:, -1]
    self.all_labels   = self.get_labels(cfg.label_dir)
    self.all_features = self.get_fetaures(cfg.features_dir) 
    self.train_nodes  = np.load(cfg.label_fold_dir+'train_ids.npy')
    self.val_nodes    = np.load(cfg.label_fold_dir+'val_ids.npy')
    self.test_nodes   = np.load(cfg.label_fold_dir+'test_ids.npy')
    #[!!!IMP!!]Assert no overlap

    self.label_cache  = self.get_label_cache(self.all_labels, self.train_nodes)
    self.update_cache = {}

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
    #z[0] =1
    all_labels = np.concatenate(([z], all_labels), 0)
    
    return all_labels

  def get_label_cache(self, all_labels, nodes):
    #dict -> {node : (label, update_count)}
    temp = {idx+1:all_labels[idx+1] for idx in np.where(nodes)}
    temp[0] = all_labels[0]
    return temp

  def accumulate_label_cache(self, labels, nodes):
    default = (self.all_labels[0], 1)
    for idx, node in enumerate(nodes):
      prv_label, prv_count = self.update_cache.get(node, default)
      new_label = prv_label + labels[idx]
      new_count = prv_count + 1
      self.update_cache[node] = (new_label, new_count)

  def update_label_cache(self):
    #{!!!IMP} Assert original training labels are not updated
    for k,v in self.update_cache.items():
      assert self.train_nodes[k-1] is False
      
      label = v[0]/v[1] #average of all predictions for this label
      self.label_cache[k] = label
      
    self.update_cache = {} #reset the update cache


  def get_nodes(self, dataset):
    if dataset == 'train':
        nodes = self.train_nodes
    elif dataset == 'val':
        nodes = self.val_nodes
    elif dataset == 'test':
        nodes = self.test_nodes
    elif dataset == 'all':
        #Get all the nodes except the training nodes
        nodes = np.logical_not(self.train_nodes)

    return nodes
    
  def next_batch(self, dataset, batch_size=None, shuffle=True):

    nodes = self.get_nodes(dataset)    
    label_len = np.shape(self.all_labels)[1]

    #Get position of all walks ending with desired set of nodes
    pos = []
    for idx in np.where(nodes):
        node = idx + 1
        pos.extend(np.where(self.node_seq == node)[0])

    if shuffle:
        indices = np.random.permutation(len(pos))
        pos = pos[indices]
        
    for i in range(0,len(pos), batch_size):
        x = self.all_walks[pos[i: i+batch_size]]        
        #convert from (batch x step) to (step x batch)
        x = np.swapaxes(x, 0,1)
        
        #get labels for valid data points, for others: select the 0th label
        #convert from (batch x step x feature_size) to (step x batch x feature_size)
        default = self.label_cache[0][0]
        x2 = [[self.label_cache.get(item, default) for item in row] for row in x] 

        y = x2[-1, :] #Labels of all the final nodes

        #get features for all data points
        x = [[features[item] for item in row] for row in x]
        
        seq = node_seq[i: i+batch_size]
                    
        yield (x, x2, seq, y)


 
