from __future__ import generators
#import arff
import collections
import numpy as np
from tensorflow.python.framework import dtypes
from Utils import ptb_iterator
from vocab import Vocab
from random import shuffle

class DataSet(object):

  def __init__(self,x,vocab,labels,dtype=dtypes.float32):
    """Construct a DataSet.
    """
    #Add vocab to the dataset
    self.vocab = vocab
    self.labels = labels
    dtype = dtypes.as_dtype(dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
      raise TypeError('Invalid image dtype %r, expected uint8 or float32' %
                      dtype)
    self._x = x

  def next_batch(self, batch_size, num_steps,shuffle=True):
    """Return the next `batch_size` examples from this data set.
       Takes Y in one-hot encodings only"""
    for x,y,y2,seq_len in ptb_iterator(self._x, batch_size, num_steps,shuffle, self.labels):
        yield x,y,y2,seq_len 

def get_ptb_dataset(dataset):
    for line in open(dataset):
      #Starting and ending is the same symbol
      yield '<eos>'
      for word in line.split():
        if word == '0':
          yield '<eos>'
        #[FIXED]: else statement missing?
        else:
          yield word
      yield '<eos>'

def get_labels(data_dir, vocab,  percent = 1):
  data = open(data_dir).read().split()
  labels = {}
  inverse_labels = {}
  reduced_labels = {}
  for i in range(0, len(data), 2):
    #Labels start with node 0; Walks_data with node 1
    #To get corresponding mapping, increment the label node number by 1
    #label: 0 is used for <eos> and nodes without labels
    n = vocab.encode(str(int(data[i])+1))
    l = int(data[i+1]) + 1
    labels[n] = labels.get(n, [])
    labels[n].append(l)

  #[TODO] FIX label removal
  #Total number of labels kept is more than 'percent' as reduced_label[v] = labels[v]
  #
  #Reduce the number of labels, if required
  if percent < 1:    
    #convert it to {label:nodes} format
    for k,val in labels.items():
      #Multilables
      for v in val:
        #[DOUBT] 'inverse_labels[v] = inverse_labels.get(v, []).append(k)' returns NONE???
        inverse_labels[v] = inverse_labels.get(v, [])
        inverse_labels[v].append(k)
      
    #keep only 'percentage' of nodes randomly
    for k,val in inverse_labels.items():
      shuffle(val)
      count = int(len(val)*percent)
      #Keep at least one node per label
      if count > 0:
        del val[count:]
      inverse_labels[k] = val

    #Convert it to {node:label} format
    for k,val in inverse_labels.items():
      for v in val:
        reduced_labels[v] = labels[v]
        
    return reduced_labels
  return labels


def read_data_sets(data_dir, label_percent, dtype=dtypes.float32,validation_ratio=0.20):
    vocab = Vocab()
    vocab.construct(get_ptb_dataset(data_dir+'p_walks.txt'))
    train_x = np.array([vocab.encode(word) for word in get_ptb_dataset(data_dir+'train_walks.txt')],dtype=np.int32)
    validation_x = np.array([vocab.encode(word) for word in get_ptb_dataset(data_dir+'val_walks.txt')],dtype=np.int32)

    #Consider only a percentage of labels for training
    labels = get_labels(data_dir+'labels.txt', vocab, label_percent)
    #Consider all labels for validating
    val_labels = get_labels(data_dir+'labels.txt',vocab, 1)
    
    train = DataSet(train_x, vocab,labels, dtype=dtype)
    validation = DataSet(validation_x, vocab, val_labels, dtype=dtype)

    datasets_template = collections.namedtuple('Datasets_template', ['train','validation'])
    Datasets = datasets_template(train=train,validation=validation)

    return Datasets

def load_ptb_data():
  return read_data_sets(train_dir=None)

#IMPROVEMENTS REQUIRED
# ADD LINK CHECKING
