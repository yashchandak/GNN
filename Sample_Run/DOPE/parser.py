import argparse


class Parser(object):

    def __init__(self):
            parser = argparse.ArgumentParser()

            parser.add_argument("--path", default='/home/priyesh/Desktop/Expt_Deep_CC',
                                help="Base path for the code")
            parser.add_argument("--project", default='DOPE', help="Project folder")
            parser.add_argument("--folder_suffix", default='Default', help="folder name suffix")
            parser.add_argument("--run_test", default=False, type=self.str2bool, help="Run test at every inner fit")

            parser.add_argument("--dataset", default='citeseer', help="Dataset to evluate")
            parser.add_argument("--labels", default='labels_random', help="Label type")
            parser.add_argument("--percents", default='20', help="Training percent")
            parser.add_argument("--folds", default='1_2_3_4_5', help="Training folds")
            parser.add_argument("--max_walks", default=0, help="Maximum No of walks | 0 - sample by degree", type=int)
            parser.add_argument("--max_depth", default=1, help="Maximum path depth", type=int)

            parser.add_argument("--retrain", default=False, type=self.str2bool, help="Retrain flag")
            parser.add_argument("--debug", default=False, type=self.str2bool, help="Debug flag")

            parser.add_argument("--batch_size", default=30, help="Batch size", type=int)
            parser.add_argument("--lr", default=0.01, help="Learning rate", type=float)
            parser.add_argument("--lu", default=0.75, help="Label update rate", type=float)
            parser.add_argument("--l2", default=1e-2, help="L2 loss", type=float)
            parser.add_argument("--opt", default='adam', help="Optimizer type (adam, rmsprop, sgd)")

            parser.add_argument("--pat", default=2, help="Patience", type=int)
            parser.add_argument("--pat_inc", default=2, help="Patience Increase", type=int)
            parser.add_argument("--pat_improve", default=1, help="Improvement threshold for patience", type=float)
            parser.add_argument("--save_after", default=0, help="Save after epochs", type=int)
            parser.add_argument("--val_freq", default=1, help="Validation frequency", type=int)

            parser.add_argument("--bin_upd", default=0, help="Binary updates for labels", type=int)
            parser.add_argument("--gradients", default=0, help="Print gradients of trainable variables", type=int)
            parser.add_argument("--max_outer", default=1, help="Maximum outer epoch", type=int)
            parser.add_argument("--max_inner", default=200, help="Maximum inner epoch", type=int)

            parser.add_argument("--boot_epochs", default=1, help="Epochs for first bootstrap", type=int)
            parser.add_argument("--boot_reset", default=1, help="Reset weights after first bootstrap", type=int)
            parser.add_argument("--concat", default=0, help="Concat attribute to hidden state", type=int)
            parser.add_argument("--drop_in", default=0.5, help="Dropout for input", type=float)
            parser.add_argument("--drop_out", default=0.6, help="Dropout for pre-final layer", type=float)

            parser.add_argument("--wce", default=0, help="Weighted cross entropy", type=int)
            parser.add_argument("--attention", default=1, help="Attention module (0: no, 1: HwC, 2: tanh(wH + wC))",
                                type=int)
            parser.add_argument("--ssl", default=0, help="Semi-supervised loss", type=int)
            parser.add_argument("--inner_converge", default=0, help="Convergence during bootstrap", type=int)

            parser.add_argument("--cell", default='myRNN', help="RNN cell (LSTM, myLSTM, GRU, LSTMgated)")
            parser.add_argument("--reduce", default=0, help="Reduce Attribute dimensions to", type=int)
            parser.add_argument("--hidden", default=16, help="Hidden units", type=int)

            parser.add_argument("--node_loss", default=0, help="Node Loss", type=int)
            parser.add_argument("--path_loss", default=0, help="Path Loss", type=int)
            parser.add_argument("--consensus_loss", default=0, help="Consensus Loss", type=int)


            self.parser = parser

    def str2bool(self, text):
        if text == 'True':
            arg = True
        elif text == 'False':
            arg = False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')
        return arg

    def get_parser(self):
        return self.parser

