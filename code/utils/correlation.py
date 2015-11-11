#Calculate the correlations between each time point

import numpy as np
import matplotlib.pyplot as plt
from stimuli import events2neural
import nibabel as nib

def correlations(fname, tr):
	img = nib.load(fname + ".nii")
	TR = tr
	n_vol = img.shape[-1]

	time_course = events2neural(fname + "_cond.txt", TR, n_vol)
	time_course = time_course[4:]
	data = img.get_data()[...,4:]

	n_voxels = np.prod(data.shape[:-1])
	data_2d = np.reshape(data, (n_voxels, data.shape[-1]))
	correlations_1d = np.zeros((n_voxels, ))

	for i in range(n_voxels):
		correlations_1d[i] = np.corrcoef(time_course, data_2d[i, :])[0,1]

	correlations = np.reshape(correlations_1d, data.shape[:-1])

	return correlations

if __name__ == '__main__':
	from sys import argv

	filename = argv[1]

	# calling 
	# python correlation.py ds114_sub009_t2r1
	corr_mat = correlations(filename, 2.5)
	plt.imshow(corr_mat[:, :, 14])
	plt.show()
