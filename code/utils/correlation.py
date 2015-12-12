import numpy as np  # the Python array package
import nibabel as nib
import matplotlib.pyplot as plt  # the Python plotting package
import pearson
import stimuli

def read_image(fname):
	img = nib.load('../../data/sub001/BOLD/' + fname + '/filtered_func_data_mni.nii.gz')
	data = img.get_data()[...,4:]

	plt.figure()
	fig, axes = plt.subplots(nrows=4, ncols=4)
	for i, ax in zip(range(25,58,2), axes.flat):
		im = ax.imshow(data[:, i, :, 65], cmap='gray', alpha=0.5)
	fig.subplots_adjust(right=0.85)
	cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
	fig.suptitle("Middle Level for brain", fontsize=20)
	fig.colorbar(im, cax=cax)
	plt.savefig("../../data/sample_image.png")

	return data

def faster_correlation(fname, data):
	n_trs = data.shape[-1]

	# Get on-off timecourse
	time_course = stimuli.events2neural('../../data/sub001/onsets' + f1 + '_conv002.txt', 2.5, n_trs)

	# Calculate the number of voxels (number of elements in one volume)
	n_voxels = np.prod(data.shape[:-1])

	# Reshape 4D array to 2D array n_voxels by n_volumes
	data_2d = np.reshape(data, (n_voxels, data.shape[-1]))
	
	# Transpose 2D array to n_trs x n_vols
	data_2d_T = data_2d.T

	# Calculate 1D vector length n_voxels of correlation coefficients
	correlation_1d = pearson.pearson_2d(time_course, data_2d_T)

	# Reshape the correlations array back to 3D
	correlation_3d = correlation_1d.reshape(data.shape[:-1])

	# Plot middle level correlation
	fig, axes = plt.subplots(nrows=4, ncols=4)
	for i, ax in zip(range(25,58,2), axes.flat):
		im = ax.imshow(correlation_3d[:, i, :, 65], cmap='gray', alpha=0.5)
	fig.subplots_adjust(right=0.85)
	cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
	fig.suptitle("Middle Level of correlations for brain", fontsize=20)
	fig.colorbar(im, cax=cax)
	plt.savefig("../../data/middle_correlation.png")	

if __name__ == '__main__':
	from sys import argv

	f1 = argv[1]
	data = read_image(f1)
	correlations = faster_correlation(f1, data)

	plt.show()