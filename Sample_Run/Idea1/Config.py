import tensorflow as tf
import sys, os, shutil

class Config(object):


    codebase_root_path = '/home/priyesh/Desktop/Codes/Sample_Run/'
    sys.path.insert(0, codebase_root_path)

    ####  Directory paths ####
    #Folder name and project name is the same
    project_name = 'Idea1'
    dataset_name = 'BlogDWdata'
   
    logs_d   = '/Logs/'
    ckpt_d   = '/Checkpoints/'
    embed_d  = '/Embeddings/'
    result_d = '/Results/'

    #Retrain
    retrain = True

    #Debug with small dataset
    debug = False

    # Batch size
    batch_size = 128
    num_steps = 7
    #Number of steps to run trainer
    max_epochs = 1000
    #Validation frequence
    val_epochs_freq = 1
    #Model save frequency
    save_epochs_after= 0

    #Other hyperparameters
    #thresholding constant
    th=0.4

    #earlystopping hyperparametrs
    patience = max_epochs # look as this many epochs regardless
    patience_increase = 2 # wait this much longer when a new best is found
    improvement_threshold = 0.9999  # a relative improvement of this much is considered significant

    metrics = ['coverage','average_precision','ranking_loss','micro_f1','macro_f1','micro_precision',
               'macro_precision','micro_recall','macro_recall','p@1','p@3','p@5','hamming_loss']
    
    def __init__(self):
        self.init2()

    def init2(self):
        self.train_dir = self.codebase_root_path + 'Datasets/' + self.dataset_name+'/'

        #Logs and checkpoints to be stored in the code directory
        self.project_prefix_path = self.codebase_root_path+ self.project_name+'/'


    def check_n_create(self, path):
        if not os.path.exists(path):
            os.mkdir(path)
        else:
            if not self.retrain:
                shutil.rmtree(path)
                os.mkdir(path)
        
    def create(self, ext_path =""):
        #create directories
        ext_path = './'+ext_path
        self.logs_dir = ext_path + self.logs_d
        self.ckpt_dir = ext_path + self.ckpt_d
        self.embed_dir= ext_path + self.embed_d
        self.result_dir = ext_path+self.result_d

        self.check_n_create(ext_path)
        self.check_n_create(self.logs_dir)
        self.check_n_create(self.ckpt_dir)
        self.check_n_create(self.embed_dir)
        self.check_n_create(self.result_dir)
        
    class Solver(object):
        def __init__(self):
            self._parameters = {}
            #Initial learning rate
            self._parameters['learning_rate'] = 0.001

            #optimizer
            self._parameters['optimizer'] = tf.train.AdamOptimizer(self._parameters['learning_rate'])
            self._next_node_loss = True
            self._curr_label_loss = True
            self._label_similarity_loss = False
            self._embedding_loss = False
            
    class Architecture(object):
        def __init__(self):
            self._parameters = {}
            #Number of layer - excluding the input & output layers
            self._parameters['num_layers'] = 2
            #Mention the number of layers
            self._parameters['layers'] = [100,250]
            #dropout
            self._dropout = 0.9

    class Data_sets(object):
        def __init__(self):
            self._len_vocab = 0
            self._len_labels = 0
            self._diffusion_rate = 0.75
            self._emb_factor = 1000
            self._keep_label_percent = 1
    

    class RNNArchitecture(object):
        def __init__(self):
            #self._num_steps = 10 # Problem with reusing variable
            self._embed_size = 128
            self._hidden_size = 128
            self._dropout = 0.9
            self._layers = 1

    solver = Solver()
    architecture = Architecture()
    data_sets = Data_sets()
    mRNN = RNNArchitecture()

    # TO DO
    #http://stackoverflow.com/questions/33703624/how-does-tf-app-run-work


