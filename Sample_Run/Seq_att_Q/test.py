import tensorflow as tf
import numpy as np
import threading
import time
batch_size =32
flag = False


class CustomRunner(object):
    """
    This class manages the the background threads needed to fill
        a queue full of data.
    """
    def __init__(self):

        #self.data_iterator = data
        self.n_threads = 2
        self.dataX = tf.placeholder(dtype=tf.float32, shape=[None, 10])
        self.dataY = tf.placeholder(dtype=tf.int64, shape=[None, ])
        # The actual queue of data. The queue contains a vector for
        # the mnist features, and a scalar label.

        self.queue =  tf.RandomShuffleQueue(shapes=[[10], []],
                      dtypes=[tf.float32, tf.int64],
                      capacity=100,
                      min_after_dequeue=0)

        # The symbolic operation to add data to the queue
        # we could do some preprocessing here or do it in numpy. In this example
        # we do the scaling in numpy
        self.enqueue_op = self.queue.enqueue_many([self.dataX, self.dataY])

        #self.qr = tf.train.QueueRunner(self.queue, [self.enqueue_op] * 4)

    def data_iterator(self, n):
        #global flag
        # while True:
        print('===============TRAIN==========')
        total_samples = 1024
        samples_per_thread = total_samples//self.n_threads
        start = n*samples_per_thread
        for idx in range(start, start + samples_per_thread, batch_size):
            feats = np.random.randn(batch_size, 10)
            labels = np.random.randint(0, 2, batch_size)
            yield feats, labels
        return

    def data_iterator2(self, n):
        #global flag
        # while True:
        print('===============VAL==========')
        total_samples = 1024
        samples_per_thread = total_samples//self.n_threads
        start = n*samples_per_thread
        for idx in range(start, start + samples_per_thread, batch_size):
            feats = np.ones((batch_size, 10))
            labels = np.random.randint(0, 2, batch_size)
            yield feats, labels
        return

    def get_inputs(self):
        """
        Return's tensors containing a batch of images and labels
        """
        images_batch, labels_batch = self.queue.dequeue_many(batch_size)
        return images_batch, labels_batch

    def thread_main(self, sess, coord, n, samples='train'):
        """
        Function run on alternate thread. Basically, keep adding data to the queue.
        """
        if samples == 'train': fn = self.data_iterator
        elif samples == 'val': fn = self.data_iterator2

        for dataX, dataY in fn(n):
            sess.run(self.enqueue_op, feed_dict={self.dataX:dataX, self.dataY:dataY})

        #exit()
        #coord.request_stop() #Non-deterministic, can make other threads stop early
        #sess.run(self.queue.close(cancel_pending_enqueues=False))

    def start_threads(self, sess, coord, samples):
        """ Start background threads to feed queue """
        threads = []
        for n in range(self.n_threads):
            t = threading.Thread(target=self.thread_main, args=(sess, coord, n, samples))
            t.daemon = True # thread will close when parent quits
            t.start()
            threads.append(t)
        return threads


#Doing anything with data on the CPU is generally a good idea.
with tf.device("/cpu:0"):
    custom_runner = CustomRunner()
    images_batch, labels_batch = custom_runner.get_inputs()

# simple model
w = tf.get_variable("w1", [10, 2])
y_pred = tf.matmul(images_batch, w)
sum = tf.reduce_sum(images_batch)
loss = tf.nn.sparse_softmax_cross_entropy_with_logits(y_pred, labels_batch)

# for monitoring
loss_mean = tf.reduce_mean(loss)
train_op = tf.train.AdamOptimizer().minimize(loss)

sess = tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=8))
init = tf.initialize_all_variables()
sess.run(init)

def step(samples):
    coord = tf.train.Coordinator()
    # start the tensorflow QueueRunner's
    #threads1 = tf.train.start_queue_runners(sess=sess, coord=coord)
    # start our custom queue runner's threads
    threads = custom_runner.start_threads(sess, coord, samples=samples)
    print "%d Threads started"%len(threads)

    ctr = 0
    #One epoch
    try:
        while True :
            _, sum_val, loss_val = sess.run([train_op, sum, loss_mean])
            print "ctr %d : "%ctr, loss_val, sum_val, sess.run(custom_runner.queue.size()), threading.activeCount()
            ctr += 1

            #if coord.should_stop() and sess.run(custom_runner.queue.size()) == 0: break
            if threading.active_count() == 1  and sess.run(custom_runner.queue.size()) == 0: break

    except Exception, e:
        print('catched exception: ', e)
        coord.request_stop(e)
    #finally:
    #coord.request_stop()
    #coord.join(threads)
start = time.time()
for i in range(3):
    step('val')
    print("==================", i)
step('val')
print time.time()-start