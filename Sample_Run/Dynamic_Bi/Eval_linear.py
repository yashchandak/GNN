from __future__ import print_function

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

import scipy
import numpy

import Eval_Calculate_Performance as perf
from Eval_Data import Data
from Eval_Config import Config
import Eval_utils as utils


def get_dense(inp, size):
    dense = numpy.zeros((len(inp),size))
    for i in range(len(inp)):
        dense[i][inp[i]] = 1
    return dense

def evaluate(cfg):
        data = Data(cfg)
        all_results = {}
        for train_percent in cfg.training_percents:
            all_results[train_percent] = {}
            for shuf in range(cfg.num_shuffles):
                data.set_training_validation(('train',shuf, int(train_percent*100)), ('valid',shuf, int(train_percent*100)))

                X_train, Y_train = data.get_training_sparse()
                X_test, Y_test   = data.get_validation_sparse()

                Y_train_dense = get_dense(Y_train, cfg.label_len)
                Y_test_dense  = get_dense(Y_test, cfg.label_len)

                clf = OneVsRestClassifier(LogisticRegression())
                clf.fit(X_train, scipy.sparse.coo_matrix(Y_train_dense))

                best_th  = 0
                if cfg.threshold:
                    best_val, i = -1, 0.1
                    while(i<0.3):
                        preds = clf.predict_proba(X_train)
                        val = perf.evaluate(preds, Y_train_dense, threshold=i)[3] #3 = micr0-f1, 4=macro-f1
                        if val > best_val:
                            best_th = i 
                            best_val = val        
                        i += 0.1

                    print("best th: ", best_th)    

		
                preds = clf.predict_proba(X_test)
                results = perf.evaluate(preds, Y_test_dense, best_th)
                all_results[train_percent][shuf] = results  


        for train_percent in sorted(all_results.keys()):
            print ('Train percent:', train_percent)
            micro, macro = [], []
            #print numpy.mean(all_results[train_percent])
            x = all_results[train_percent]
            for v in x.values():
                micro.append(v[3])
                macro.append(v[4])
            print (x.values())
            print ("Micro: ",numpy.mean(micro), "  Macro: ",numpy.mean(macro))
            print ('-------------------')
            utils.write_results(cfg, all_results)

if __name__ == "__main__":
     con = Config()
     evaluate(con)

