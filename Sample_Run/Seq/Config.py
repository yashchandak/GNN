import tensorflow as tf
import sys, os, shutil

class Config(object):


    codebase_root_path = '/home/priyesh/Desktop/Codes/Sample_Run/'
    sys.path.insert(0, codebase_root_path)

    ####  Directory paths ####
    #Folder name and project name is the same
    project_name = 'Seq'
    dataset_name = 'cora'
    train_percent = 2
    train_fold  = 1
    
    logs_d   = '/Logs/'
    ckpt_d   = '/Checkpoints/'
    embed_d  = '/Embeddings/'
    result_d = '/Results/'

    #Retrain
    retrain = True

    #Debug with small dataset
    debug = False

    # Batch size
    batch_size = 32
    #Number of steps to run trainer
    max_outer_epochs = 100
    max_inner_epochs = 3
    #Validation frequence
    val_epochs_freq = 1
    #Model save frequency
    save_epochs_after= 0

    #earlystopping hyperparametrs
    patience = max_outer_epochs # look as this many epochs regardless
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.9999  # a relative improvement of this much is considered significant

    metrics = ['coverage','average_precision','ranking_loss','micro_f1','macro_f1','micro_precision',
               'macro_precision','micro_recall','macro_recall','p@1','p@3','p@5','hamming_loss','cross-entropy','accuracy']
    
    def __init__(self):
        self.init2()

    def init2(self):
        self.walks_dir = self.codebase_root_path + 'Datasets/' + self.dataset_name+'/walks/walks_80.txt'
        self.label_fold_dir = self.codebase_root_path + 'Datasets/' + self.dataset_name+'/labels/'+ str(self.train_percent) + '/' + str(self.train_fold) + '/'
        self.label_dir = self.codebase_root_path + 'Datasets/' + self.dataset_name+'/labels.npy'
        self.features_dir = self.codebase_root_path + 'Datasets/' + self.dataset_name+'/features.npy'
        #Logs and checkpoints to be stored in the code directory
        self.project_prefix_path = self.codebase_root_path+ self.project_name+'/'


    def check_n_create(self, path):
        if not os.path.exists(path):
            #if the path doesn't exists, create it
            os.mkdir(path)
        else:
            if not self.retrain:
            #path exists but if retrain in False
            #then replace previous folder with new folder
                shutil.rmtree(path)
                os.mkdir(path)
        
    def create(self, ext_path =""):
        #create directories
        ext_path = './'+ext_path
        self.logs_dir = ext_path + self.logs_d
        self.ckpt_dir = ext_path + self.ckpt_d
        #self.embed_dir= ext_path + self.embed_d
        #self.result_dir = ext_path+self.result_d

        self.check_n_create(ext_path)
        self.check_n_create(self.logs_dir)
        self.check_n_create(self.ckpt_dir)
        #self.check_n_create(self.embed_dir)
        #self.check_n_create(self.result_dir)
        
    class Solver(object):
        def __init__(self):
            #Initial learning rate
            self.learning_rate = 0.001

            #optimizer
            self._optimizer = tf.train.AdamOptimizer(self.learning_rate)
            self._curr_label_loss = True
            self._L2loss = True
            

    class Data_sets(object):
        def __init__(self):
            self._len_vocab = 0
            self._len_labels = 0
            self._len_features =0
            self.binary_label_updates = False

    class RNNArchitecture(object):
        def __init__(self):
            self._hidden_size = 16
            self._keep_prob_in = 0.7
            self._keep_prob_out = 0.7

    solver = Solver()
    data_sets = Data_sets()
    mRNN = RNNArchitecture()


