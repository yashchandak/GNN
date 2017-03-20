import numpy as np
import matplotlib.pyplot as plt


def plotit(y, fig, xlabel, ylabel, title, cfg):
    x = np.arange(len(y.values()[0]))
    for k,v in y.items():
        plt.figure(fig)
        plt.plot(x, np.flipud(v))
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    plt.savefig(cfg.results_folder+title+'.png')

  
def write_results(cfg, all_results):
    for percent, results in all_results.items():
           f = open(cfg.results_folder+str(percent)+'.txt','w')

           for metric in cfg.metrics:
                f.write(metric+ '\t')
           f.write('\n')

           arr = np.zeros((len(results.values()[0]), len(results)))#[[]]*len(results.values()[0])
           for shuff, vals in results.items():
                for idx, val in enumerate(vals):
                     arr[idx][shuff-1] = val
                     f.write(str(val) + '\t')
                f.write('\n') 

           #f.write('\n')
           for v in range(arr.shape[0]):
                f.write(str(np.mean(arr[v][:]))+ '\t')

           f.write('\n')
           for v in range(arr.shape[0]):
                f.write(str(np.var(arr[v][:]))+ '\t')
                              
           f.close()

