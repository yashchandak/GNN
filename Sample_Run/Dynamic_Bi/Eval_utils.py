import numpy as np

  
def write_results(cfg, all_results):
    for percent, results in all_results.items():
           f = open(cfg.results_folder+str(percent)+'.txt','w')

           for metric in cfg.metrics:
                f.write(metric+ '\t')
           f.write('\n')

           arr = np.zeros((len(results.values()[0]), cfg.num_shuffles))#[[]]*len(results.values()[0])
           for shuff, vals in results.items():
                for idx, val in enumerate(vals):
                     arr[idx][shuff-1] = val
                     f.write(str(val) + '\t')
                f.write('\n') 

           f.write('\n')
           for v in range(arr.shape[0]):
                f.write(str(np.mean(arr[v][:]))+ '\t')

           f.write('\n')
           for v in range(arr.shape[0]):
                f.write(str(np.var(arr[v][:]))+ '\t')
                              
           f.close()

