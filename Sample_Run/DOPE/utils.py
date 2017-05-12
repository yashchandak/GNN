from os import path, mkdir
import shutil
import tensorflow as tf
import numpy as np

def check_n_create(dir_path, overwrite=False):
    if not path.exists(dir_path):
        mkdir(dir_path)
    else:
        if overwrite:
            shutil.rmtree(dir_path)
            mkdir(dir_path)


def create_directory_tree(dir_path):
    for i in range(len(dir_path)):
        check_n_create(path.join(*(dir_path[:i + 1])))


#SET SEED FOR ALL
def relu_init(shape, dtype=tf.float32, partition_info=None):
    init_range = np.sqrt(2.0/(shape[1]))
    initial = tf.random_normal(shape,dtype=dtype) * init_range
    return initial

def zeros(shape, dtype=tf.float32, partition_info=None):
    """All zeros."""
    initial = tf.zeros(shape, dtype=dtype)
    return initial

def const(shape, dtype=tf.float32, partition_info=None):
    initial = tf.constant(0.1, shape=shape, dtype=dtype)
    return initial


def tanh_init(shape, dtype=tf.float32, partition_info=None):
    init_range = np.sqrt(6.0/(shape[0]+shape[1]))
    initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
    return initial