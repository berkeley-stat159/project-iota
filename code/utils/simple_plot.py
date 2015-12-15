import numpy as np  # the Python array package
import nibabel as nib
import matplotlib.pyplot as plt  # the Python plotting package
import pearson
import stimuli

def read_image(fname):
	img = nib.load('../../data/sub001/BOLD/' + fname + '/filtered_func_data_mni.nii.gz')
	data = img.get_data()[...,4:]

	fig = plt.figure()
	a = fig.add_subplot(1,3,1)
	imgplot = plt.imshow(data[45, :, :, 66], cmap='gray', alpha=0.5)
	a.set_title("Front Level for brain")
	a = fig.add_subplot(1,3,2)
	imgplot = plt.imshow(data[:, 50, :, 66], cmap='gray', alpha=0.5)
	a.set_title("Middle Level for brain")
	a = fig.add_subplot(1,3,3)
	plt.imshow(data[:, :, 45, 66], cmap='gray', alpha=0.5)
	a.set_title("Back Level for brain")
	plt.colorbar()
	plt.savefig("../../data/sample_image.png")

	return data

if __name__ == '__main__':
	from sys import argv

	f1 = argv[1]
	data = read_image(f1)
	
