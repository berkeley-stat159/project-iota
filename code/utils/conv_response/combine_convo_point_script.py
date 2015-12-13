import matplotlib.pyplot as plt
import numpy as np
from sys import argv

f1 = argv[1] # task001_run001

block_convo = np.array([])
full_convo = np.array([])

block_num = [1,4,5]
full_num = range(1,7,1)

"""
block_list = ['task001_run001/cond001.txt', 'task001_run001/cond004.txt', 'task001_run001dconv005.txt']
"""
block_list = []
for i in block_num:
    block_list.append(f1 + '/cond00' + str(i) + '.txt')

"""
full_list = ['task001_run001/cond001.txt', 'task001_run001/cond002.txt', 'task001_run001/cond003.txt', 'task001_run001/cond004.txt', 
'task001_run001/cond005.txt', 'task001_run001/cond006.txt']
"""
full_list = []
for i in full_num:
    full_list.append(f1 + '/cond00' + str(i) + '.txt')

for i in block_list:
	block_convo = np.append(block_convo, np.loadtxt('../../../data/sub001/onsets/' + i))

for i in full_list:
	full_convo = np.append(full_convo, np.loadtxt('../../../data/sub001/onsets/' + i))

block_time = block_convo[range(0, len(block_convo), 3)]
block_val = block_convo[range(2, len(block_convo), 3)]

full_time = full_convo[range(0, len(full_convo), 3)]
full_val = full_convo[range(2, len(full_convo), 3)]

plt.figure(0)
plt.plot(block_time, block_val, '.')
plt.xlabel('Experiment Time')
plt.ylabel('Study condition amplitude')
plt.title('Block Model Condition')
plt.savefig('../../../data/convo/' + f1 + '_block_points.png')

plt.figure(1)
plt.plot(full_time, full_val, '.')
plt.xlabel('Experiment Time')
plt.ylabel('Study condition amplitude')
plt.title('Full Model Condition')
plt.savefig('../../../data/convo/' + f1 + '_full_points.png')