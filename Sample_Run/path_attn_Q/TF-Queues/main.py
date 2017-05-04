from __future__ import print_function
import tensorflow as tf
import numpy as np
from dataset import Dataset
import threading


data_dir = 'citeseer'
ds = Dataset(data_dir, max_diameter=10)
mixed_batch_size = 1
mixed_total_batches = np.ceil(ds.n_nodes/mixed_batch_size)

mixed_walks_queue = tf.FIFOQueue(capacity=10, dtypes=[tf.int32, tf.int32, tf.int32, tf.int32])
mixed_batch_placeholder = tf.placeholder(tf.int32, shape=[1, None, ds.diameter+1])
random_placeholder = tf.placeholder(tf.int32, shape=[1, 10])
random_placeholder2 = tf.placeholder(tf.int32, shape= [1])
random_placeholder3 = tf.placeholder(tf.int32, shape= [1,2,5, 20])


enqueue_op = mixed_walks_queue.enqueue_many([mixed_batch_placeholder, random_placeholder2, random_placeholder, random_placeholder3])
d1, d2, d3, d4 = mixed_walks_queue.dequeue()
#op = batch

with tf.Session() as sess:

    def load_and_enqueue():
        for batch_id, data in enumerate(ds.get_batches(mixed_batch_size, mixed_total_batches,
                                                       ds.mixed_walks_generator(n_walks=1, by_max_degree=True,
                                                                                by_prob=True))):
          print("raw: ",data)
          #data = data[0]
          #data = np.expand_dims(np.array(data), axis=0)
          sess.run(enqueue_op, feed_dict={mixed_batch_placeholder: data, random_placeholder:np.ones((1,10)),  random_placeholder2:[5], random_placeholder3:np.ones((1,2,5,20))})

    t = threading.Thread(target=load_and_enqueue)
    t.daemon = True
    t.start()

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(coord=coord)

    for i in range(10):
        data1, data2, data3, data4 = sess.run([d1, d2, d3, d4])
        print("DATA: ",data1, data2, data3, data4)
        #print(i, len(data), data[0].shape)
        #print(data[0][0][:])

    coord.request_stop()
    coord.join(threads)
    #t._stop()



