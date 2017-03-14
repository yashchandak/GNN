from __future__ import print_function

from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression

import scipy
import numpy

import Eval_Calculate_Performance as perf
from Eval_Data import Data
from Eval_Config import Config
import Eval_utils as utils


def evaluate(cfg):
        data = Data(cfg)
        all_results = {}
        for train_percent in cfg.training_percents:
            all_results[train_percent] = {}
            for shuf in cfg.num_shuffles:
                #set the data for current train percent and fold
                data.set_training_validation(train_percent, shuf)

                X_train, Y_train_dense = data.get_all_nodes_labels('train')
                X_test, Y_test_dense   = data.get_all_nodes_labels('test')
                #print(data.get_all_nodes_labels('test'))

                #Fit a linear classifier on the data
                clf = OneVsRestClassifier(LogisticRegression())
                #clf.fit(X_train, scipy.sparse.coo_matrix(Y_train_dense))
                clf.fit(X_train, Y_train_dense)

                best_th  = 0
                if cfg.threshold:
                    best_val, i = -1, 0.1
                    #do grid search on validation set to find best threshold value
                    while(i<0.3):
                        preds = clf.predict_proba(X_train)
                        val = perf.evaluate(preds, Y_train_dense, threshold=i)[-1] #3 = micr0-f1, 4=macro-f1 #-1=accuracy
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
            micro, acc = [], []
            x = all_results[train_percent]
            for v in x.values():
                micro.append(v[3])
                acc.append(v[-1])
            print (x.values())
            print ("Micro: ",numpy.mean(micro), "  Accuracy: ",numpy.mean(acc))
            print ('-------------------')

        utils.write_results(cfg, all_results)

if __name__ == "__main__":
    con  = Config('cora/','/home/priyesh/Desktop/Codes/Sample_Run/Datasets/cora/features.npy')
    evaluate(con)

