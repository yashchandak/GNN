from __future__ import print_function
import os.path
import time, math, sys
from copy import deepcopy
import scipy.sparse as sps
import numpy as np
from sklearn.preprocessing import normalize

import tensorflow as tf
from tensorflow.contrib.tensorboard.plugins import projector
import blogDWdata as input_data
import network as architecture
import Config as conf
import Eval_Calculate_Performance as perf
from Utils import labels_to_onehot, sample
from copy import deepcopy

#import Eval_MLP as NN
import Eval_linear as liblinear
import Eval_Config

cfg = conf.Config() 


#Code structure inspired from Stanford's cs224d assignment starter codes
#class DNN(Model):
class RNNLM_v1(object):
    def __init__(self, config):
        self.config = config
        # Generate placeholders for the images and labels.
        self.load_data()
        self.add_placeholders()
        #self.add_metrics()

        # Build model
        self.arch = self.add_network(config)
        self.inputs = self.arch.embedding(self.data_placeholder)
        self.rnn_outputs = self.arch.predict(self.inputs,self.keep_prob, self.seq_len)
        self.outputs = self.arch.projection(self.rnn_outputs)

        # casting to handle numerical stability
        self.predictions_next = [tf.nn.softmax(tf.cast(o, 'float64')) for o in self.outputs[0]]
        # Reshape the output into len(vocab) sized chunks - the -1 says as many as
        # needed to evenly divide
        output_next = tf.reshape(tf.concat(1, self.outputs[0]), [-1, self.config.data_sets._len_vocab])
        #output_label = tf.reshape(tf.concat(1, self.outputs[1]), [-1, self.config.data_sets._len_labels])
        output_label =  self.outputs[1]
        
        self.loss = self.arch.loss([output_next, output_label], self.label_placeholder, self.label_2_placeholder, self.inputs)
        self.optimizer = self.config.solver._parameters['optimizer']
        self.train = self.arch.training(self.loss,self.optimizer)

        self.saver = tf.train.Saver(write_version=tf.train.SaverDef.V2)
        self.summary = tf.summary.merge_all()
        self.step_incr_op = self.arch.global_step.assign(self.arch.global_step+1)
        #local variable initialization required for metrics operation, otherwise throws error
        # self.init = tf.group(tf.initialize_all_variables(), tf.initialize_local_variables())
        self.init = tf.global_variables_initializer()#tf.initialize_all_variables()
        
    def load_data(self):
        # Get the 'encoded data'
        self.data_sets =  input_data.read_data_sets(self.config)
        debug = self.config.debug
        if debug:
            print('##############--------- Debug mode ')
            num_debug = (self.config.num_steps+1)*128
            self.data_sets.train._x = self.data_sets.train._x[:num_debug]
            self.data_sets.validation._x  = self.data_sets.validation._x[:num_debug]
            #self.data_sets.test_x  = self.data_sets.test_x[:num_debug]
        
        self.config.data_sets._len_vocab = self.data_sets.train.vocab.__len__()

        l = len(list(self.data_sets.train.labels.values())[0])
        self.config.data_sets._len_labels= l

        print('--------- Project Path: '+self.config.codebase_root_path+self.config.project_name)
        print('--------- Vocabulary Length: '+str(self.config.data_sets._len_vocab))
        print('--------- Label Length: '+str(self.config.data_sets._len_labels))
        print('--------- No. of Labelled nodes: ' + str(len(self.data_sets.train.labels.keys())))

    def add_placeholders(self):
        #self.data_placeholder = tf.placeholder(tf.int32,shape=[None,self.config.num_steps],name='Input')
	#self.label_placeholder = tf.placeholder(tf.int32,shape=[None,self.config.num_steps],name='Target')
        self.data_placeholder = tf.placeholder(tf.int32,shape=[None,self.config.num_steps], name='Input')
        self.label_placeholder = tf.placeholder(tf.int32,name='Target')
        self.label_2_placeholder = tf.placeholder(tf.int32,name='Target_label')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.seq_len = tf.placeholder(tf.int32, shape=[None], name='Seq_len')
    	#self.metrics = tf.placeholder(tf.float32,shape=(len(self.config.metrics),))

    def create_feed_dict(self, input_batch, label_batch, label_batch_2, seq_len):
        feed_dict = {
            self.data_placeholder: input_batch,
            self.label_placeholder: label_batch,
            self.label_2_placeholder: label_batch_2,
            self.seq_len: seq_len
        }
        return feed_dict

    def add_network(self, config):
        return architecture.Network(config)

    def add_metrics(self, metrics):
        """assign and add summary to a metric tensor"""
        for i,metric in enumerate(self.config.metrics):
            tf.summary.scalar(metric, metrics[i])

    def add_summaries(self,sess):
        # Instantiate a SummaryWriter to output summaries and the Graph.
        self.summary_writer_train = tf.train.SummaryWriter(self.config.logs_dir+"train", sess.graph)
        self.summary_writer_val = tf.train.SummaryWriter(self.config.logs_dir+"val", sess.graph)
    
    def write_summary(self,sess,summary_writer, metric_values, step, feed_dict):
        summary = self.summary.merged_summary
        #feed_dict[self.loss]=loss
        feed_dict[self.metrics]=metric_values
        summary_str = sess.run(summary, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)
        summary_writer.flush()


    def run_epoch(self, sess, dataset, train_op=None, summary_writer=None,verbose=1000):
        if not train_op :
            train_op = tf.no_op()
            keep_prob = 1
        else:
            keep_prob = self.config.architecture._dropout
        # And then after everything is built, start the training loop.
        total_loss = []
        next_loss = []
        label_loss = []
        sim_loss = []
        emb_loss = []
        total_steps = sum(1 for x in dataset.next_batch(self.config.batch_size,self.config.num_steps))	
	#Sets to state to zero for a new epoch
        state = self.arch.initial_state.eval()
        for step, (input_batch, label_batch, label_batch_2, seq_len) in enumerate(
            dataset.next_batch(self.config.batch_size,self.config.num_steps)):

            #print("\n\n\nActualLabelCount: ", input_batch, label_batch, label_batch_2, seq_len, np.sum(label_batch_2, axis=2))
            feed_dict = self.create_feed_dict(input_batch, label_batch, label_batch_2, seq_len)
            feed_dict[self.keep_prob] = keep_prob
	    #Set's the initial_state temporarily to the previous final state for the session  "AWESOME" -- verified
	    #feed_dict[self.arch.initial_state] = state 
	    
	    #Writes loss summary @last step of the epoch
            if (step+1) < total_steps:
                _, loss_value, state, pred_labels = sess.run([train_op, self.loss, self.arch.final_state, self.arch.label_sigmoid], feed_dict=feed_dict)
            else:
                _, loss_value, state, summary, pred_labels = sess.run([train_op, self.loss, self.arch.final_state,self.summary,self.arch.label_sigmoid], feed_dict=feed_dict)
                if summary_writer != None:
                    summary_writer.add_summary(summary,self.arch.global_step.eval(session=sess))
                    summary_writer.flush()
            #print(loss_value)
            total_loss.append(loss_value[0])
            next_loss.append(loss_value[1])
            label_loss.append(loss_value[2])
            sim_loss.append(loss_value[3])
            emb_loss.append(loss_value[4])

            #print("\n\n\nPredLabels:", pred_labels)

            if verbose and step % verbose == 0:
                metrics = [0]*20
                if self.config.solver._curr_label_loss:
                    metrics = perf.evaluate(pred_labels, label_batch_2, 0)
                    self.add_metrics(metrics)
                sys.stdout.write('\r{} / {} : pp = {} : next = {} : label = {} : micro-F1  =  {} : macro-F1 = {} : sim = {} : emb = {}'.format(step, total_steps, np.exp(np.mean(total_loss)), np.mean(next_loss), np.mean(label_loss), metrics[3], metrics[4], np.mean(sim_loss), np.mean(emb_loss)))
                sys.stdout.flush()
            
        if verbose:
            sys.stdout.write('\r')
        return np.exp(np.mean(total_loss)),np.mean(total_loss)

    def fit(self, sess):
        #define parametrs for early stopping early stopping
        max_epochs = self.config.max_epochs
        patience = self.config.patience  # look as this many examples regardless
        patience_increase = self.config.patience_increase  # wait this much longer when a new best is found
        improvement_threshold = self.config.improvement_threshold  # a relative improvement of this much is
                                                             # considered significant
        
        # go through this many minibatches before checking the network on the validation set
        # Here we check every epoch
        validation_loss = 1e6
        done_looping = False
        step = 1
        best_step = -1
        losses = []
        learning_rate = self.config.solver._parameters['learning_rate']
        #sess.run(self.init) #DO NOT DO THIS!! Doesn't restart from checkpoint
        while (step <= self.config.max_epochs) and (not done_looping):
            #print 'Epoch {}'.format(epoch)
	    #step_incr_op = tf.assign_add(self.global_step,1)
            sess.run([self.step_incr_op])
            epoch = self.arch.global_step.eval(session=sess)

            start_time = time.time()
            tr_pp, average_loss = self.run_epoch(sess,self.data_sets.train,train_op=self.train,summary_writer=self.summary_writer_train)
            duration = time.time() - start_time

            if (epoch % self.config.val_epochs_freq == 0):
                val_pp,val_loss = self.run_epoch(sess,self.data_sets.validation,summary_writer=self.summary_writer_val)

                print('\n Epoch %d: tr_loss = %.2f, val_loss = %.2f || tr_pp = %.2f, val_pp = %.2f  (%.3f sec)'
                      % (epoch, average_loss, val_loss, tr_pp, val_pp, duration))
                	
                # Save model only if the improvement is significant
                if (val_loss < validation_loss * improvement_threshold) and (epoch > self.config.save_epochs_after):
                    patience = max(patience, epoch * patience_increase)
                    validation_loss = val_loss
                    checkpoint_file = self.config.ckpt_dir + 'checkpoint'
                    self.saver.save(sess, checkpoint_file, global_step=epoch)
                    best_step = epoch
                    patience = epoch + max(self.config.val_epochs_freq,self.config.patience_increase)
                    print('best step %d'%(best_step))
		
                elif val_loss > validation_loss * improvement_threshold:
                    patience = epoch - 1

            else:
		    # Print status to stdout.
                print('Epoch %d: loss = %.2f pp = %.2f (%.3f sec)' % (epoch, average_loss, tr_pp, duration))

            if (patience <= epoch):
		#config.val_epochs_freq = 2
                learning_rate = learning_rate / 10
                self.optimizer = tf.train.AdamOptimizer(learning_rate)
                patience = epoch + max(self.config.val_epochs_freq,self.config.patience_increase)
                print('--------- Learning rate dropped to: %f'%(learning_rate))		
                if learning_rate <= 0.0000001:
                    print('Stopping by patience method')
                    done_looping = True

            losses.append(average_loss) 
            step += 1

        return losses, best_step

    def get_embedding(self,sess,data, layer = 0):
        if layer == 0:
            feed_dict = {self.data_placeholder: [data], self.keep_prob: 1, self.arch.initial_state: self.arch.initial_state.eval()}
            return sess.run(self.inputs,feed_dict=feed_dict)[0]
	
        if layer == 1:
            feed_dict = {self.data_placeholder: [data], self.keep_prob: 1, self.arch.initial_state: self.arch.initial_state.eval(), self.seq_len:[1]}
            return sess.run(self.rnn_outputs, feed_dict=feed_dict)[0]

        else:
            print("Undefined layer")
            return

    def get_hidden_state(self,sess,data,eos_embed=None):
        if eos_embed is None:
           eos_embed = self.arch.initial_state.eval()
        feed_dict = {self.data_placeholder: [data], self.keep_prob: 1, self.arch.initial_state: eos_embed, self.seq_len:[1]}
        return sess.run(self.rnn_outputs,feed_dict=feed_dict)[0]

    def generate_text(self,session, starting_text='<eos>',stop_length=100, stop_tokens=None, temp=1.0 ):
        """Generate text from the model.
	  Args:
	    session: tf.Session() object
	    starting_text: Initial text passed to model.
	  Returns:
	    output: List of word idxs
	"""
        state = self.arch.initial_state.eval()
	# Imagine tokens as a batch size of one, length of len(tokens[0])
        tokens = [self.data_sets.train.vocab.encode(word) for word in starting_text.split()]
        all_labels = []
        for i in range(stop_length):
            feed = {self.data_placeholder: [tokens[-1:]], self.arch.initial_state: state, self.keep_prob: 1}
            state, y_pred, embed, pred_labels = session.run([self.arch.final_state, self.predictions_next[-1],self.inputs, self.arch.label_sigmoid], feed_dict=feed)
            state = state[0]
            all_labels.append(pred_labels[0][0])  #batch-0, seq number-0
            next_word_idx = sample(y_pred[0], temperature=temp)
            tokens.append(next_word_idx)
            if stop_tokens and self.data_sets.train.vocab.decode(tokens[-1]) in stop_tokens:
                break
        output = [self.data_sets.train.vocab.decode(word_idx) for word_idx in tokens]

        #Print out the next nodes and corresponding labels

        #print("labels and nodes are both incremented by 1 as compared to original dataset")
        #for step, labels in enumerate(all_labels):
        #    temp = []
        #    for idx, val in enumerate(labels):
        #        if val>0.25:
        #            temp.append(idx)
        #    print(output[step], ": ", temp)

        return output
    
    #def generate_sentence(self,session,starting_text,temp):  
    def generate_sentence(self,session,*args, **kwargs):
        """Convenice to generate a sentence from the model."""
        return self.generate_text(session, *args, stop_tokens=['<eos>'], **kwargs)



########END OF CLASS MODEL#############################################################################################################

def init_Model(config):
    tf.reset_default_graph()
    with tf.variable_scope('RNNLM',reuse=None) as scope:
        model = RNNLM_v1(config)
    
    tfconfig = tf.ConfigProto( allow_soft_placement=True)
    tfconfig.gpu_options.allow_growth = True
    sm = tf.train.SessionManager()

    if config.retrain:
        load_ckpt_dir = config.ckpt_dir
        print('--------- Loading variables from checkpoint if available')
    else:
        load_ckpt_dir = ''
        print('--------- Training from scratch')
    sess = sm.prepare_session("", init_op=model.init, saver=model.saver, checkpoint_dir=load_ckpt_dir,config=tfconfig)
    return model, sess

def train_DNNModel():
    #global cfg
    print('############## Training Module ')
    config = deepcopy(cfg)
    model,sess = init_Model(config)
    with sess:
	    model.add_summaries(sess)
	    losses, best_step = model.fit(sess)
    return losses

def test_DNNModel():
    #global cfg
    print('############## Test Module ')
    config = deepcopy(cfg)
    model,sess = init_Model(config)    
    with sess:
        test_pp = model.run_epoch(sess,model.data_sets.validation)
        print('=-=' * 5)
        print('Test perplexity: {}'.format(test_pp))
        print('=-=' * 5)

def interactive_generate_text_DNNModel():
    #global cfg
    print('############## Generate Text Module ')
    config = deepcopy(cfg)
    config.batch_size = config.num_steps = 1
    model,sess = init_Model(config)
    with sess:
        starting_text = '2'
        while starting_text:
          print(' '.join(model.generate_sentence(sess, starting_text=starting_text, temp=1.0)))
          starting_text = input('> ')

def dump_generate_text_DNNModel():
    global cfg
    print('############## Generate sentences for all words in dictionary and Dump  ')
    config = deepcopy(cfg)
    config.batch_size = config.num_steps = 1
    model,sess = init_Model(config)
    num_sentences = 2
    with sess:
        ignore_list = ['0','<eos>','<unk>'] 
        keys = [int(word) for word in model.data_sets.train.vocab.word_freq.keys() if word not in ignore_list] 
        keys.sort()
        vocab_len = len(keys)
        f_id = config.dataset_name+'/_data.sentences','w'

        for starting_text in keys:
            for n in range(num_sentences):
                words = model.generate_sentence(sess, starting_text=str(starting_text), temp=1.0)
                f_id.write((' '.join(words[:-1])+'\n'))



def save_Embeddings_DNNModel():
    #global cfg
    print('############## Save Embeddings Module ')
    config = deepcopy(cfg)
    config.batch_size = config.num_steps = 1
    model,sess = init_Model(config)
    with sess:
        model.add_summaries(sess)
        ignore_list = ['0','<eos>','<unk>'] 
        keys = [int(word) for word in model.data_sets.train.vocab.word_freq.keys() if word not in ignore_list] 
        keys.sort()
        vocab_len = len(keys)
        enc_words = np.array([model.data_sets.train.vocab.encode(str(word)) for word in keys])
        #embed = np.zeros([vocab_len,model.config.mRNN._embed_size])
        embed = np.zeros([vocab_len,model.config.mRNN._hidden_size])

        #eos_embed = model.get_embedding(sess,['<eos>'])
        eos_embed = model.get_hidden_state(sess,[model.data_sets.train.vocab.encode('<eos>')],None)

        for i,word in enumerate(enc_words):
            embed[i] = model.get_embedding(sess,[word],)
            #embed[i] = model.get_hidden_state(sess,[word],eos_embed)

        fn = config.embed_dir+config.dataset_name+'_data.embd'
        np.savetxt(fn,embed, delimiter=',')
        #np.savetxt(fn,normalize(embed,norm='l2',axis=1), delimiter=',')
        print('--------- Embeddings are saved to '+fn)

        
def save_Embeddings_DNNModel2(): #UNUSED
    #global cfg
    print('############## Save Embeddings Module ')
    config = deepcopy(cfg)
    config.batch_size = config.num_steps = 1
    model,sess = init_Model(config)
    with sess:
        model.add_summaries(sess)
        ignore_list = ['0','<eos>','<unk>'] 
        keys = [int(word) for word in model.data_sets.train.vocab.word_freq.keys() if word not in ignore_list] 
        keys.sort()
        vocab_len = len(keys)
        enc_words = np.array([model.data_sets.train.vocab.encode(str(word)) for word in keys])

        embed0 = np.zeros([vocab_len,model.config.mRNN._embed_size])
        embed1 = np.zeros([vocab_len,model.config.mRNN._hidden_size])
        for i,word in enumerate(enc_words):
            embed0[i] = model.get_embedding(sess,[word], layer=0)
            embed1[i] = model.get_embedding(sess,[word], layer=1)
	
                
        fn0 = config.embed_dir+config.dataset_name+'_data0.embd'
        fn1 = config.embed_dir+config.dataset_name+'_data1.embd'

        save_embed(config.embed_dir+config.dataset_name+'_data0'+str(config.mRNN._embed_size)+'.embd', embed0)
        save_embed(config.embed_dir+config.dataset_name+'_data1'+str(config.mRNN._hidden_size)+'.embd', embed1)

        np.savetxt(fn0, embed0, delimiter=',')
        np.savetxt(fn1, embed1, delimiter=',')

        print('--------- Embeddings are saved to '+fn0)

def save_embed(path, embed): #UNUSED
	f = open(path, 'w')
	for idx, item in enumerate(embed):
		f.write(str(idx))
		for val in item:
			f.write(' ' + str(val))
		f. write('\n')
	f.close()

def visualize_Embeddings_DNNModel():
    #global cfg
    print('############## Visualize Embeddings Module ')
    config = deepcopy(cfg)
    tf.reset_default_graph()
    sess = tf.Session()
    fn = config.embed_dir+config.dataset_name+'_data.embd'
    #fn = config.embed_dir+'karate_structure_features'
    print('--------- Embeddings are loaded from dir: '+fn)
    embed = np.loadtxt(fn,delimiter=',')
    embed_var = tf.Variable(embed,name='embed_var')
    init = tf.initialize_all_variables()
    sess.run(init)

    checkpoint_file = config.logs_dir, 'Embedding'
    saver = tf.train.Saver({"embedding": embed_var},write_version=tf.train.SaverDef.V2)
    fn = config.embed_dir+'embedding_ckpt'
    saver.save(sess,fn, global_step=1)
    print('--------- To Visualize Embeddings load tf:0.12v tensorboard in directory: '+fn)


def generate_and_reconstruct():
    print('############## Reconstruct Text Module ')
    config = deepcopy(cfg)
    config.batch_size = config.num_steps = 1
    model,sess = init_Model(config)

    ignore_list = ['0','<eos>','<unk>'] 
    keys =  [word for word in model.data_sets.train.vocab.word_freq.keys() if word not in ignore_list]
    nodes = len(keys)
    #adj_mat = np.zeros((nodes, nodes), dtype=int)
    adj_list = {}
    walk_count = 10
    
    with sess:
        for idx, node in enumerate(keys):
            if idx%100 == 0:
                print("Reconstructing for node: ",idx)
            for i in range(walk_count):   
                walk = model.generate_sentence(sess, starting_text=node, temp=1.0)
                for n1, n2 in zip(walk[:-2], walk[1:-1]):
                    #Subtracting one to start node count from 0
                    n1, n2 = int(n1)-1, int(n2)-1
                    weight = adj_list.get((n1, n2), 0)
                    adj_list[(n1,n2)] = weight+1
                    #adj_mat[int(n1)-1][int(n2)-1] += 1

    adj_mat = sps.lil_matrix((nodes, nodes))
    for k, v in adj_list.items():
         i,j = k
         adj_mat[i,j] = v

    #adj_mat = scipy.sparse.coo_matrix(adj_mat)
    savemat(config.results_dir+'reconstructed_'+cfg.dataset_name, adj_mat)
    print('------------ Reconstruction file saved: ', 'reconstructed_'+cfg.dataset_name )

def classify_and_save():
    print('############## Classify and save Module ')
    config = deepcopy(cfg)
    fn = config.embed_dir+config.dataset_name+'_data.embd'

    e_conf = Eval_Config.Config(config.dataset_name+'/', fn)
    #NN.evaluate(e_conf)
    liblinear.evaluate(e_conf)
    print("------------ Results saved to: ", e_conf.results_folder)    


def execute():
    with tf.device('/gpu:0'):
        #err = train_DNNModel() 
        #test_DNNModel() 
        #interactive_generate_text_DNNModel()
        #save_Embeddings_DNNModel()
        #visualize_Embeddings_DNNModel() 
        #generate_and_reconstruct()
        classify_and_save()  
        return err
    
if __name__ == "__main__":
    #deepcopy cfg
    #remove parameter dictionary
    #Embeddings and checkpoint directorty
    
    meta_param = {('dataset_name',):['blogcatalog_ncc'],
                  ('solver', 'learning_rate'): [0.001],
                  ('retrain',): [True],
                  ('debug',): [False],
                  ('max_epochs',): [1000]
    }

    variations = len(meta_param[('dataset_name',)])

    #Make sure number of variants are equal
    for k,v in meta_param.items():
        assert len(v) == variations           
    
    
    for idx in range(variations):        
        for k,vals in meta_param.items():
            x = cfg
            if len(k) > 1:
                x = getattr(x, k[0])
            setattr(x, k[-1], vals[idx])
            print(k[-1], vals[idx])

        cfg.create(cfg.dataset_name)#"run-"+str(idx))
        cfg.init2()

        #All set... GO! 
        execute()
        print('\n\n ===================== \n\n')
