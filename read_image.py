#read in bold.nii.gz
#./project-iota/sub001/BOLD/task001_run001

import nibabel as nib
import numpy as num
import matplotlib.pyplot as plt
#%matplotlib

img = nib.load("bold.nii.gz")
data = img.get_data()

vol0 = data[...,0]

plt.imshow(vol0[...,0], interpolation="nearest")