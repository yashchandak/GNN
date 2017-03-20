import tensorflow as tf
import sys, os, shutil

class Config():

    max_epochs = 10000 #Number of steps to run trainer
    val_epochs_freq = 25  #Validation frequence    
    save_epochs_after= 1 #Model save frequency
    retrain = True
    #earlystopping hyperparametrs
    patience = max_epochs # look as this many epochs regardless
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.9999  # a relative improvement of this much is considered significant
    
    def __init__(self, dataset, embed_file):
        self.root_path = '/home/priyesh/Desktop/Codes/Sample_Run/'
        self.project_name = 'Seq/'
        self.logs_d = '/Logs/'


        self.dataset = dataset
        self.embed_file = embed_file

        self.input_len = 1433
        self.label_len = 7
        
        self.hidden = 0
        self.batch_size = 10
        self.learning_rate = 0.1
        self.drop = 1

        self.metrics = ['coverage','average_precision','ranking_loss','micro_f1','macro_f1','micro_precision',
                        'macro_precision','micro_recall','macro_recall','p@1','p@3','p@5','hamming_loss','cross_entropy', 'accuracy']
        self.training_percents = [25]
        self.threshold = False
        self.num_shuffles = [1,2,3,4,5]
	
        self.init2()

    def init2(self):
        #self.optimizer   = tf.train.RMSPropOptimizer(self.learning_rate)#(self.learning_rate)
        #self.optimizer   = tf.train.AdamOptimizer(self.learning_rate)
        self.optimizer   = tf.train.GradientDescentOptimizer(self.learning_rate)

        self.directory   = self.root_path + 'Datasets/' + self.dataset
        self.label_dir  = self.directory + 'labels.npy'
        self.results_folder = self.root_path + self.project_name + self.dataset + 'Results/'

    def check_n_create(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        #else:
            #if not self.retrain:
            #    shutil.rmtree(path)
            #os.mkdir(path)

        
    def create(self, ext_path =""):
        #create directories
        ext_path = './'+ext_path
        self.logs_dir = ext_path + self.logs_d
        self.check_n_create(ext_path)
        self.check_n_create(self.logs_dir)
        self.check_n_create(self.results_folder)
