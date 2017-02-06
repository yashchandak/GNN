from __future__ import generators
#import arff
import collections
import numpy as np
from tensorflow.python.framework import dtypes
from Utils import ptb_iterator
from vocab import Vocab
from random import shuffle
from scipy.io import loadmat

class DataSet(object):

  def __init__(self,x,vocab,labels,features,dtype=dtypes.float32):
    """Construct a DataSet.
    """
    #Add vocab to the dataset
    self.labels = labels
    self.features = features
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    self._x = x

  def next_batch(self, batch_size, num_steps,shuffle=True):
    """Return the next `batch_size` examples from this data set.
       Takes Y in one-hot encodings only"""
    for x,y,y2,seq_len in ptb_iterator(self._x, batch_size, num_steps,shuffle, self.labels, self.features):
        yield x,y,y2,seq_len
 
def get_ptb_dataset(dataset):
    for line in open(dataset):
      #Starting and ending is the same symbol
      yield 0
      for word in line.split():
          yield int(word)
      yield 0


def get_labels(all_labels, nodes):
    #Labels start with node '0'; Walks_data with node '1'
    #To get corresponding mapping, increment the label node number by 1
    #append new 0th label and shift every other label right by 1 -->
    #--> 0th label is used for <eos> and nodes without labels
    new_labels = {}
    for idx, val in enumerate(nodes):
      if val:
        new_labels[idx+1] = np.concatenate(([0],all_labels[idx]))

    return new_labels
      
    
def read_data_sets(cfg, dtype=dtypes.float32):
  
    train_x = np.array([word for word in get_ptb_dataset(cfg.train_dir+'train_walks.txt')],dtype=np.int32)
    validation_x = np.array([word for word in get_ptb_dataset(cfg.train_dir+'val_walks.txt')],dtype=np.int32)

    all_labels        = loadmat(cfg.label_dir)['labels']
    all_features      = np.load(cfg.attribute_dir)['features']
    all_features      = all_features.astype(np.int32, copy=False) #Required conversion for Python3
    label_train_nodes = np.load(cfg.label_fold_dir+'train_ids.npy')
    label_val_nodes   = np.load(cfg.label_fold_dir+'test_ids.npy')

    train_labels      = get_labels(all_labels, label_train_nodes)
    val_labels        = get_labels(all_labels, label_val_nodes  )

    train             = DataSet(train_x, train_labels, all_features, dtype=dtype)
    validation        = DataSet(validation_x, val_labels, all_features, dtype=dtype)

    datasets_template = collections.namedtuple('Datasets_template', ['train','validation'])
    Datasets          = datasets_template(train=train,validation=validation)

    return Datasets

def load_ptb_data():
  return read_data_sets(train_dir=None)

#IMPROVEMENTS REQUIRED
# ADD LINK CHECKING
