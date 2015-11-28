from __future__ import division
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import linear_modeling

"""
run python linear_modeling_script  sub001/BOLD/task001_run001/filtered_func_data_mni
"""
# set gray colormap and nearest neighbor interpolation by default
plt.rcParams['image.cmap'] = 'gray'
plt.rcParams['image.interpolation'] = 'nearest'

# Load f1, the BOLD image.
f1 = argv[1] # sub001/BOLD/task001_run001/bold
# or use the filtered data: sub001/BOLD/task001_run001/filtered_func_data_mni
img = nib.load('../../../data/' + f1 + '.nii.gz')
data = img.get_data()

# Get n_trs
n_trs = data.shape[-1]

# we take the mean volume (over time), and do a histogram of the values
mean_vol = np.mean(data, axis=-1)
plt.hist(np.ravel(mean_vol), bins=100)
plt.xlabel('Voxels')
plt.ylabel('Frequency')
plt.title('Mean Volume Over Time')
plt.savefig("../../data/design_matrix/mean_vol.png")

# mask out the outer-brain noise using mean volumes over time.
in_brain_mask = mean_vol > 5000

# Load the convolution files.
design_mat = np.ones((n_trs, 8))
files = ('1', '2', '3', '4', '5', '6')
for file in files:
    design_mat[:, (int(file) - 1)] = np.loadtxt('data/convo_prep/task001_run001_cond00'
                                          + file + '_conv.txt')
# adding the linear drifter terms to the design matrix
linear_drift = np.linspace(-1, 1, n_trs)
design_mat[:, 6] = linear_drift
quadratic_drift = linear_drift ** 2
quadratic_drift -= np.mean(quadratic_drift)
design_mat[:, 7] = quadratic_drift

plt.imshow(design_mat, aspect=0.1)
plt.savefig('data/design_matrix/design_mat.png')



# We can use this 3D mask to index into our 4D dataset.
# This selects all the voxel time-courses for voxels within the brain
# (as defined by the mask)

in_brain_tcs = data[in_brain_mask, :]

Y = in_brain_tcs.T
B = npl.pinv(X).dot(Y)

b_vols = np.zeros(vol_shape + (4,))
b_vols[in_brain_mask, :] = B.T

plt.imshow(b_vols[:, :, 14, 0])
plt.imshow(b_vols[:, :, 14, 1])
plt.imshow(b_vols[:, :, 14, 2])
plt.imshow(b_vols[:, :, 14, 3])
