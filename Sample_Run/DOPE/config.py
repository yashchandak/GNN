import tensorflow as tf
import sys
from os import path, mkdir
import utils


class Config(object):
    def __init__(self, args):
        self.codebase_root_path = args.path
        self.folder_suffix = args.folder_suffix
        sys.path.insert(0, self.codebase_root_path)

        ####  Directory paths ####
        # Folder name and project name is the same
        self.project_name = args.project
        self.dataset_name = args.dataset
        self.train_percent = args.percents
        self.train_fold = args.folds

        self.logs_d = 'Logs/'
        self.ckpt_d = 'Checkpoints/'
        self.embed_d = 'Embeddings/'
        self.result_d = 'Results/'

        # Retrain
        self.retrain = args.retrain
        # Debug with small dataset
        self.debug = args.debug
        # Run test
        self.run_test = args.run_test

        # Batch size
        self.batch_size = args.batch_size
        # maximum depth for trajecory from NOI
        self.max_depth = args.max_depth
        self.max_walks = args.max_walks
        # Number of steps to run trainer
        self.max_outer_epochs = args.max_outer
        self.max_inner_epochs = args.max_inner
        self.boot_epochs = args.boot_epochs
        self.boot_reset = args.boot_reset
        # Validation frequence
        self.val_epochs_freq = args.val_freq  # 1
        # Model save frequency
        self.save_epochs_after = args.save_after  # 0

        # earlystopping hyperparametrs
        self.patience = args.pat  # look as this many epochs regardless
        self.patience_increase = args.pat_inc  # wait this much longer when a new best is found
        self.improvement_threshold = args.pat_improve  # a relative improvement of this much is considered significant

        self.metrics = ['coverage', 'average_precision', 'ranking_loss', 'micro_f1', 'macro_f1', 'micro_precision',
                        'macro_precision', 'micro_recall', 'macro_recall', 'p@1', 'p@3', 'p@5', 'hamming_loss',
                        'bae', 'cross-entropy', 'accuracy']

        class Solver(object):
            def __init__(self, t_args):
                # Initial learning rate
                self.learning_rate = t_args.lr
                self.label_update_rate = t_args.lu

                # optimizer
                if args.opt == 'adam':
                    self.opt = tf.train.AdamOptimizer
                elif args.opt == 'rmsprop':
                    self.opt = tf.train.RMSPropOptimizer
                elif args.opt == 'sgd':
                    self.opt = tf.train.GradientDescentOptimizer
                else:
                    raise ValueError('Undefined type of optmizer')

                self._optimizer = self.opt
                self._curr_label_loss = True
                self._L2loss = t_args.l2
                self.wce = t_args.wce
                self.gradients = t_args.gradients

                self.node_loss = t_args.node_loss
                self.path_loss = t_args.path_loss
                self.consensus_loss = t_args.consensus_loss

        class Data_sets(object):
            def __init__(self, t_args):
                self.reduced_dims = t_args.reduce
                self.binary_label_updates = t_args.bin_upd
                self.label_type = t_args.labels

        class RNNArchitecture(object):
            def __init__(self, t_args):
                self._hidden_size = t_args.hidden
                self._keep_prob_in = 1 - t_args.drop_in
                self._keep_prob_out = 1 - t_args.drop_out
                self.cell = args.cell
                self.concat = args.concat
                self.attention = args.attention

        self.solver = Solver(args)
        self.data_sets = Data_sets(args)
        self.mRNN = RNNArchitecture(args)

        self.logs_dir = ""
        self.ckpt_dir = ""
        # self.embed_dir= "
        self.results_folder = ""
        self.project_path, self.walks_dir, self.fold_dir, self.label_path, self.features_path, \
            self.length_path, self.project_prefix_path = self.set_paths()

    def set_paths(self):
        # Logs and checkpoints to be stored in the code directory
        project_prefix_path = path.join(self.codebase_root_path, self.project_name, self.folder_suffix)

        project_path = path.join(self.codebase_root_path, 'Datasets', self.dataset_name)
        walks_dir = path.join(project_path, 'walks/walks_80.txt')
        fold_dir = path.join(path.join(project_path, self.data_sets.label_type,
                                       str(self.train_percent), str(self.train_fold)))
        label_path = path.join(path.join(project_path, 'labels.npy'))
        features_path = path.join(path.join(project_path, 'features.npy'))
        length_path = path.join(path.join(project_path, 'length.pkl'))

        return project_path, walks_dir, fold_dir, label_path, features_path, length_path, project_prefix_path

    def create_directories(self, ext_path):
        print(ext_path)
        self.logs_dir = path.join(ext_path, self.logs_d)
        self.ckpt_dir = path.join(ext_path, self.ckpt_d)
        # self.embed_dir= path.join(self.dataset_name, ext_path, self.embed_d)
        self.results_folder = path.join(ext_path, self.result_d)

        utils.check_n_create(self.logs_dir, overwrite=not self.retrain)
        utils.check_n_create(self.ckpt_dir, overwrite=not self.retrain)
        # self.check_n_create(self.embed_dir, overwrite=not self.retrain)
        utils.check_n_create(self.results_folder, overwrite=not self.retrain)

        self.project_path, self.walks_dir, self.fold_dir, self.label_path, self.features_path, \
            self.length_path, self.project_prefix_path = self.set_paths()
