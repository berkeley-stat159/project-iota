#read in bold.nii.gz
#./project-iota/sub001/BOLD/task001_run001

import nibabel as nib
import numpy as num
import matplotlib.pyplot as plt

def read_image(fname):
	img = nib.load("bold.nii.gz")
	data = img.get_data()

	vol0 = data[...,0]
	plt.imshow(vol0[...,0], interpolation="nearest")
	plt.show()

if __name__ == '__main__':
	from sys import argv

	filename = argv[1]
	read_image(filename)