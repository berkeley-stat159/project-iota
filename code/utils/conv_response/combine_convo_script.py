import matplotlib.pyplot as plt
import numpy as np
from sys import argv

block_convo = np.zeros((137,))
full_convo = np.zeros((137,))

block_list = ['task001_run001_conv001.txt', 'task001_run001_conv004.txt', 'task001_run001_conv005.txt']
full_list = ['task001_run001_conv001.txt', 'task001_run001_conv002.txt', 'task001_run001_conv003.txt', 'task001_run001_conv004.txt', 'task001_run001_conv005.txt', 'task001_run001_conv006.txt']

for i in block_list:
	block_convo = block_convo + np.loadtxt('../../../data/convo/' + i)

for i in full_list:
	full_convo = full_convo + np.loadtxt('../../../data/convo/' + i)

plt.figure(0)
plt.plot(block_convo)
plt.title("Combination of block convo response")
plt.savefig('../../../data/convo/combine_block_convo.png')

plt.figure(1)
plt.plot(full_convo)
plt.title("Combination of all convo response")
plt.savefig('../../../data/convo/combine_all_convo.png')
plt.show()