#read in bold.nii.gz
#./project-iota/sub001/BOLD/task001_run001

import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt

def read_image(fname):
	img = nib.load('../../data/' + fname + '.nii.gz')
	data = img.get_data()[...,4:]

	vol0 = data[...,65]
	plt.imshow(vol0[...,45], interpolation="nearest", cmap = 'gray')
	plt.colorbar()
        plt.title('task001_bold_mcf_brain[:,:,45,65]')
	plt.savefig("sample_image.png")

if __name__ == '__main__':
	from sys import argv

	filename = argv[1]
	read_image(filename)
