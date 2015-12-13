from __future__ import division
from sys import argv
import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import convo_response

# Load f1, the BOLD image.
f1 = argv[1] # task001_run001
img = nib.load('../../../data/sub001/BOLD/' + f1 + '/filtered_func_data_mni.nii.gz')
data = img.get_data()

# Get n_trs
n_trs = data.shape[-1]

# For loop to generate all convo-reponse
full_num = range(1,7,1)

for i in full_num:
	cond_data = np.loadtxt('../../../data/sub001/onsets/' + f1 + '/cond00' + str(i) + '.txt')
	getname = f1 + '/cond00' + str(i)
	getname = getname.replace('/', '_').replace('d', 'v')

	# Load the specified TR
	TR = 2.5
	# Specify the desired level of resolution
	tr_div = 100

	# Use rescale_con function to get the high resolution time and neural prediction
	# time course
	high_res_neural, high_res_times = convo_response.rescale_cond(cond_data, n_trs,
	                                                              TR, tr_div)
	# Plot the high resolution convolved values
	plt.plot(high_res_times, high_res_neural)
	plt.xlabel('Time (seconds)')
	plt.ylabel('High resolution neural prediction')
	plt.savefig('../../../data/convo/High_resolution_neural_' + getname + '.png')

	# Use construction_conv function to get the convolved high resolution
	# hemodynamic response.
	high_res_hemo, high_res_times = convo_response.constructing_convo(high_res_neural,
	                                                  high_res_times)
	# Write the result to a text file.
	np.savetxt('../../../data/convo/' + getname + '.txt', high_res_hemo)
	plt.plot(high_res_times, high_res_hemo)
	plt.xlabel('Time (Seconds)')
	plt.savefig('../../../data/convo/High_resolution_' + getname + '.png')
