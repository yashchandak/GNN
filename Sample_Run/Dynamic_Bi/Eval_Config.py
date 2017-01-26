import tensorflow as tf
import sys, os, shutil

class Config():

    max_epochs = 200 #Number of steps to run trainer
    val_epochs_freq = 25  #Validation frequence    
    save_epochs_after= 0 #Model save frequency
    retrain = False
    #earlystopping hyperparametrs
    patience = max_epochs # look as this many epochs regardless
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.9999  # a relative improvement of this much is considered significant
    
    def __init__(self, dataset = 'BlogDWdata/', embed_file = '/home/priyesh/Desktop/Codes/LINE-master/linux/Embeddings/youtube_line2_128.embd'):
        self.root_path = '/home/priyesh/Desktop/Codes/Sample_Run/'
        self.project_name = 'Idea1/'
        self.logs_d = '/Logs/'


        self.dataset = dataset
        self.embed_file = embed_file

        self.input_len = 128
        self.label_len = 39
        
        self.hidden = 256

        self.batch_size = 32
        self.learning_rate = 0.001
        #self.epoch = 10000        
        self.drop = 0.8

        self.metrics = ['coverage','average_precision','ranking_loss','micro_f1','macro_f1','micro_precision',
                        'macro_precision','micro_recall','macro_recall','p@1','p@3','p@5','hamming_loss']
        self.training_percents = [0.1, 0.5, 0.9]
        self.threshold = 0
        self.num_shuffles = 2
	
        self.init2()

    def init2(self):
        self.optimizer   = tf.train.AdamOptimizer(self.learning_rate)
        
        self.directory   = self.root_path + 'Datasets/' + self.dataset
        self.label_file  = self.directory + 'labels.txt'
        self.splits_file = self.directory + 'splits.dict.npy'
        self.results_folder = self.root_path + self.project_name + self.dataset + 'Results/'

    def check_n_create(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            #if not self.retrain:
            #    shutil.rmtree(path)
            os.mkdir(path)
        
    def create(self, ext_path =""):
        #create directories
        ext_path = './'+ext_path
        self.logs_dir = ext_path + self.logs_d
        #self.ckpt_dir = ext_path + self.ckpt_d
        #self.embed_dir= ext_path + self.embed_d

        self.check_n_create(ext_path)
        self.check_n_create(self.logs_dir)
        #self.check_n_create(self.ckpt_dir)
        #self.check_n_create(self.embed_dir)
