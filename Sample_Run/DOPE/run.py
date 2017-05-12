import sys
import itertools
import subprocess
from datetime import datetime
from dateutil.relativedelta import relativedelta
from shutil import rmtree
from os import environ, mkdir, path


def diff(t_a, t_b):
    t_diff = relativedelta(t_a, t_b)
    return '{h}h {m}m {s}s'.format(h=t_diff.hours, m=t_diff.minutes, s=t_diff.seconds)


switch_gpus = False #For multiple GPUs
n_parallel_threads = 25

# Set Hyper-parameters
hyper_params = ['dataset', 'batch_size', 'cell', 'attention', 'concat', 'node_loss', 'path_loss', 'consensus_loss']
dataset = ['cora']
batch_size = [25, 40]
cell = ['myRNN', 'myLSTM']
attention = [0, 1]
concat = [0, 1]
node_loss = [0, 1]
path_loss = [0, 1]
consensus_loss = [0, 1]

#Create Log Directory for stdout Dumps
stdout_dump_path = 'stdout_dumps'
if path.exists(stdout_dump_path ):
    rmtree(stdout_dump_path)
mkdir(stdout_dump_path)

param_values = []
this_module = sys.modules[__name__]
for hp in hyper_params:
    param_values.append(getattr(this_module, hp))
combinations = list(itertools.product(*param_values))
n_combinations = len(combinations)
print('Total no of experiments: ', n_combinations)

pids = [None] * n_combinations
f = [None] * n_combinations
last_process = False
for i, setting in enumerate(combinations):
    #Create command
    command = "python __main__.py "
    folder_suffix = ""
    for name, value in zip(hyper_params, setting):
        command += "--" + name + " " + str(value) + " "
        folder_suffix += "_"+str(value)
    command += "--" + "folder_suffix " + folder_suffix
    print(i+1, '/', n_combinations, command)

    if switch_gpus and (i % 2) == 0:
        env = dict(environ, **{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "1"})
    else:
        env = dict(environ, **{"CUDA_DEVICE_ORDER": "PCI_BUS_ID", "CUDA_VISIBLE_DEVICES": "0"})

    name = path.join(stdout_dump_path, folder_suffix)
    with open(name, 'w') as f[i]:
        pids[i] = subprocess.Popen(command.split(), env=env, stdout=f[i])
    if i == n_combinations-1:
        last_process = True
    if ((i+1) % n_parallel_threads == 0 and i >= n_parallel_threads-1) or last_process:
        if last_process and not ((i+1) % n_parallel_threads) == 0:
            n_parallel_threads = (i+1) % n_parallel_threads
        start = datetime.now()
        print('########## Waiting #############')
        for t in range(n_parallel_threads-1, -1, -1):
            pids[i-t].wait()
        end = datetime.now()
        print('########## Waiting Over######### Took', diff(end, start), 'for', n_parallel_threads, 'threads')

