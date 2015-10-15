import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib

def load_data(fname, tr):
	# Load the image as an image object
	img = nib.load(fname + '.nii')
	# Load the image data as an array
	# Drop the first 4 3D volumes from the array
	data = img.get_data()[..., 4:]
	# Load the pre-written convolved time course
	convolved = np.loadtxt(fname + '_cond.txt')[4:]
	print(data.shape, convolved.shape)
	return(data, convolved)

def reg_voxels_4d(data, convolved):
	design = np.ones((len(convolved), 2))
	design[:, 1] = convolved
	data_2d = np.reshape(data, (-1, data.shape[-1]))
	betas = npl.pinv(design).dot(data_2d.T)
	betas_4d = np.reshape(betas.T, img.shape[:-1] + (-1,))

	return betas_4d
if __name__ == '__main__':
	from sys import argv

	fname = argv[1]
	data, convolved = load_data(fname, 2.5)
	#beta_hat = reg_voxels_4d(data, convolved)
	#plt.imshow(beta_hat[:, :, 14, 0], interpolation = 'nearest', cmap = 'gray')
	#plt.show()