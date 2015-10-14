import pearson
import stimuli
import numpy as np  # the Python array package
import matplotlib.pyplot as plt  # the Python plotting package
from stimuli import events2neural
import nibabel as nib

def faster_correlation(fname, tr):
	# Load the ds114_sub009_t2r1.nii image
	img = nib.load(fname + '.nii')

	# Get the number of volumes in ds114_sub009_t2r1.nii
	n_trs = img.shape[-1]

	# Time between 3D volumes in seconds
	TR = tr

	# Get on-off timecourse
	time_course = events2neural(fname + '_cond.txt', TR, n_trs)

	# Drop the first 4 volumes, and the first 4 on-off values
	data = img.get_data()
	data = data[..., 4:]
	time_course = time_course[4:]

	# Calculate the number of voxels (number of elements in one volume)
	n_voxels = np.prod(data.shape[:-1])

	# Reshape 4D array to 2D array n_voxels by n_volumes
	data_2d = np.reshape(data, (n_voxels, data.shape[-1]))
	
	# Transpose 2D array to give n_volumes, n_voxels array
	data_2d_T = data_2d.T

	# Calculate 1D vector length n_voxels of correlation coefficients
	correlation_1d = pearson.pearson_2d(time_course, data_2d_T)

	# Reshape the correlations array back to 3D
	correlation_3d = correlation_1d.reshape(data.shape[:-1])

	return correlation_3d

if __name__ == '__main__':
	from sys import argv

	filename = argv[1]
	correlations = faster_correlation(filename, 2.5)
	plt.imshow(correlations[:, :, 14])
	plt.show()