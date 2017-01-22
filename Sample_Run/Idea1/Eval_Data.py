import numpy as np

class Data():

    def __init__(self, cfg):
        self.cfg = cfg

        self.labels     = self.get_labels(self.cfg.label_file)
        self.embeddings = self.get_embeddings(self.cfg.embed_file)
	self.splits     = np.load(self.cfg.splits_file).item()
        self.set_training_validation(train = ('train',0,10), valid = ('valid',0,10))
        self.has_more   = True
        self.index      = 0


    def set_training_validation(self,train, valid):
        self.training   = self.splits[train]
        self.validation = self.splits[valid]


    def get_labels(self, data_dir):
        f = open(data_dir, 'r')
        #f.readline()#remove meta-data in first line
        data = f.read().split()
        labels = {}
        for i in range(0, len(data), 2):
            n = int(data[i])   
            l = int(data[i+1]) 
            labels[n] = labels.get(n, [])
            labels[n].append(l)

        return labels

    def get_embeddings(self, data_dir):
        f = open(data_dir, 'r')
        f.readline()#remove meta-data in first line
        data = f.read().split()
        data = [float(item) for item in data]
        embed = {}
        for i in range(0, len(data), self.cfg.input_len + 1):
            embed[int(data[i])] = data[i+1 : i+1 + self.cfg.input_len]
            #print(data[i], data[i+1:self.cfg.input_len + 1])
        return embed

    def get_embeddings_csv(self, data_dir):
        embed = np.loadtxt(data_dir, dtype='float', delimiter=',')
        return embed
    
    def read_data(self, data_dir):
        nodes = []
        data = open(data_dir).read().split()
        return [int(item) for item in data]

    def reset(self):
        self.index = 0
        self.has_more = True

    def next_batch(self):
        inp_nodes  = np.zeros((self.cfg.batch_size, self.cfg.input_len))
        inp_labels = np.zeros((self.cfg.batch_size, self.cfg.label_len))
        if self.has_more:
            for i, val in enumerate(self.training[self.index: self.index+self.cfg.batch_size]):
               
                inp_nodes[i] = self.embeddings[int(val)]
                l = np.zeros(self.cfg.label_len, int)
                l[self.labels[val]] = 1
                inp_labels[i] = l

            self.index += self.cfg.batch_size
            
        if self.index+self.cfg.batch_size >= len(self.training):
            self.has_more = False

        return inp_nodes, inp_labels

    def get_training_sparse(self):
        inp_nodes  = np.zeros((len(self.training), self.cfg.input_len))
        inp_labels = [0]*len(self.training)
        for i,val in  enumerate(self.training):
            inp_nodes[i] = self.embeddings[val]
            inp_labels[i] = self.labels[val]

        return inp_nodes, inp_labels

    def get_validation_sparse(self):
        inp_nodes  = np.zeros((len(self.validation), self.cfg.input_len))
        inp_labels = [0]*len(self.validation)
        for i,val in  enumerate(self.validation):
            inp_nodes[i] = self.embeddings[val]
            inp_labels[i] = self.labels[val]

        return inp_nodes, inp_labels


    def get_validation(self):
        inp_nodes  = np.zeros((len(self.validation), self.cfg.input_len))
        inp_labels = np.zeros((len(self.validation), self.cfg.label_len))
        for i,val in  enumerate(self.validation):
            inp_nodes[i] = self.embeddings[val]
            l = np.zeros(self.cfg.label_len)
            l[self.labels[val]] = 1.0
            inp_labels[i] = l

        return inp_nodes, inp_labels
