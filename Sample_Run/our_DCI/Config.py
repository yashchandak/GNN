import tensorflow as tf
import sys, os, shutil

class Config(object):

    def __init__(self, args):
        self.codebase_root_path = args.path
        sys.path.insert(0, self.codebase_root_path)

        ####  Directory paths ####
        # Folder name and project name is the same
        self.project_name = args.project
        self.dataset_name = args.dataset
        self.train_percent = args.percent
        self.train_fold = args.folds

        self.logs_d = '/Logs/'
        self.ckpt_d = '/Checkpoints/'
        self.embed_d = '/Embeddings/'
        self.result_d = '/Results/'

        # Retrain
        self.retrain = args.retrain
        # Debug with small dataset
        self.debug = args.debug

        # Batch size
        self.batch_size = args.batch_size
        # maximum depth for trajecory from NOI
        self.max_depth = args.max_depth
        # Number of steps to run trainer
        self.max_outer_epochs = args.max_outer
        self.max_inner_epochs = args.max_inner
        self.boot_epochs = args.boot_epochs
        self.boot_reset = args.boot_reset
        # Validation frequence
        self.val_epochs_freq = args.val_freq #1
        # Model save frequency
        self.save_epochs_after = args.save_after #0

        # earlystopping hyperparametrs
        self.patience = args.pat  # look as this many epochs regardless
        self.patience_increase = args.pat_inc  # wait this much longer when a new best is found
        self.improvement_threshold = args.pat_improve  # a relative improvement of this much is considered significant

        self.metrics = ['coverage', 'average_precision', 'ranking_loss', 'micro_f1', 'macro_f1', 'micro_precision',
                   'macro_precision', 'micro_recall', 'macro_recall', 'p@1', 'p@3', 'p@5', 'hamming_loss',
                   'cross-entropy', 'accuracy']

        class Solver(object):
            def __init__(self, args):
                # Initial learning rate
                self.learning_rate = args.lr
                self.label_update_rate = args.lu

                # optimizer
                if args.opt == 'adam': self.opt = tf.train.AdamOptimizer
                elif args.opt == 'rmsprop': self.opt = tf.train.RMSPropOptimizer
                elif args.opt == 'sgd': self.opt= tf.train.GradientDescentOptimizer
                else: raise ValueError('Undefined type of optmizer')

                self._optimizer = self.opt(self.learning_rate)
                self._curr_label_loss = True
                self._L2loss = args.l2
                self.wce = args.wce

        class Data_sets(object):
            def __init__(self, args):
                self.reduced_dims = args.reduce
                self.binary_label_updates = args.bin_upd

        class RNNArchitecture(object):
            def __init__(self, args):
                self._hidden_size = args.hidden
                self._keep_prob_in = 1 - args.drop_in
                self._keep_prob_out = 1 - args.drop_out
                self.cell = args.cell
                self.concat = args.concat
                self.attention = args.attention

        self.solver = Solver(args)
        self.data_sets = Data_sets(args)
        self.mRNN = RNNArchitecture(args)

        self.init2()

    def init2(self):
        self.walks_dir = self.codebase_root_path + 'Datasets/' + self.dataset_name+'/walks/walks_80.txt'
        self.label_fold_dir = self.codebase_root_path + 'Datasets/' + self.dataset_name+'/labels/'+ str(self.train_percent) + '/' + str(self.train_fold) + '/'
        self.label_dir = self.codebase_root_path + 'Datasets/' + self.dataset_name+'/labels.npy'
        self.features_dir = self.codebase_root_path + 'Datasets/' + self.dataset_name+'/features.npy'
        self.adj_dir = self.codebase_root_path + 'Datasets/' + self.dataset_name+'/adjmat.mat'
        #Logs and checkpoints to be stored in the code directory
        self.project_prefix_path = self.codebase_root_path+ self.project_name+'/'


    def check_n_create(self, path):
        if not os.path.exists(path):
            #if the path doesn't exists, create it
            os.mkdir(path)
        else:
            if not self.retrain:
            #path exists but if retrain in False, then replace previous folder with new folder
                shutil.rmtree(path)
                os.mkdir(path)
        
    def create(self, ext_path =""):
        #create directories
        ext_path = './'+ext_path
        self.logs_dir = ext_path + self.logs_d
        self.ckpt_dir = ext_path + self.ckpt_d
        #self.embed_dir= ext_path + self.embed_d
        self.results_folder = ext_path+self.result_d

        self.check_n_create(ext_path)
        self.check_n_create(self.logs_dir)
        self.check_n_create(self.ckpt_dir)
        #self.check_n_create(self.embed_dir)
        self.check_n_create(self.results_folder)
        



