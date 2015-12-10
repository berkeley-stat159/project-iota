from __future__ import division
from sys import argv
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import numpy.linalg as npl
import linear_modeling
from nilearn import image
from nilearn.plotting import plot_stat_map
from scipy.ndimage import gaussian_filter

"""
set the directory to project-iota and run on terminal:
python full_linear_modeling_script.py sub001/BOLD/task001_run001/filtered_func_data_mni

or run on ipython:
python full_linear_modeling_script.py

"""
##ipython test load data:
#img = nib.load("../../../data/sub001/BOLD/task001_run001/filtered_func_data_mni.nii.gz")

############## Load f1, the BOLD image.
f1 = argv[1] #sub001/BOLD/task001_run001/filtered_func_data_mni
img = nib.load('../../../data/' + f1 + '.nii.gz')
data = img.get_data()
data = data[..., 4:]

############## Get n_trs and voxel shape
n_trs = data.shape[-1]
vol_shape = data.shape[:-1]

############## Construct design matrix:
# Load the convolution files.
design_mat = np.ones((n_trs, 9))
files = range(1,7,1)
for file in files:
    design_mat[:, (file - 1)] = np.loadtxt('../../../data/convo/task001_run001_conv00' +
                                           str(file) + '.txt')[4:]
############## adding the linear drifter terms to the design matrix
linear_drift = np.linspace(-1, 1, n_trs)
design_mat[:, 6] = linear_drift
quadratic_drift = linear_drift ** 2
quadratic_drift -= np.mean(quadratic_drift)
design_mat[:, 7] = quadratic_drift

# show the design matrix graphically:
plt.imshow(design_mat, aspect=0.1, cmap='gray', interpolation = 'nearest')
plt.savefig('../../../data/design_matrix/full_design_mat.png')
plt.close()
np.savetxt('../../../data/design_matrix/full_design_mat.txt', design_mat)


############## we take the mean volume (over time), and do a histogram of the values
mean_vol = np.mean(data, axis=-1)
plt.hist(np.ravel(mean_vol), bins=100)
plt.xlabel('Voxels')
plt.ylabel('Frequency')
plt.title('Mean Volume Over Time')
#plt.show()
#plt.savefig("../../../data/design_matrix/mean_vol.png")

# mask out the outer-brain noise using mean volumes over time.
in_brain_mask = mean_vol > 8000
# We can use this 3D mask to index into our 4D dataset.
# This selects all the voxel time-courses for voxels within the brain
# (as defined by the mask)
############## Spatially smoothing the raw data
y = linear_modeling.smoothing(data, in_brain_mask)


############## Lastly, do t test on betas:
X = design_mat

beta, MRSS, df = linear_modeling.beta_est(y,X)
print('The mean MRSS across all voxels using all 6 study conditions is ' + str(np.mean(MRSS)))

# Visualizing betas for the middle slice
# First reshape
b_vols = np.zeros(vol_shape + (beta.shape[0],))
b_vols[in_brain_mask, :] = beta.T
# Then plot them on the same plot with uniform scale
fig, axes = plt.subplots(nrows=2, ncols=4)
for i, ax in zip(range(0,8,1), axes.flat):
    im = ax.imshow(b_vols[:, :, 45, i], cmap = 'gray')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.savefig("../../../data/maps/full_beta.png")
plt.close()


############## To test significance of betas:
# Create contrast matrix for each beta:
c_mat = np.diag(np.array(np.ones((9,))))
# t statistics and p values
# Length is the number of voxels after masking
t_mat = np.ones((9, y.shape[1]))
p_mat = np.ones((9, y.shape[1],))
for i in range(0,9,1):
    t, p = linear_modeling.t_stat(X, c_mat[:,i], beta, MRSS, df)
    t_mat[i,:] = t
    p_mat[i,:] = p
# save the t values and p values in txt files.
np.savetxt('../../../data/maps/full_t_stat.txt', t_mat)
np.savetxt('../../../data/maps/full_p_val.txt', p_mat)

############## Reshape t values
t_val = np.zeros(vol_shape + (t_mat.shape[0],))
t_val[in_brain_mask, :] = t_mat.T
# Reshape p values
p_val = np.zeros(vol_shape + (p_mat.shape[0],))
p_val[in_brain_mask, :] = p_mat.T

############## Visualizing t values for the middle slice
fig, axes = plt.subplots(nrows=2, ncols=4)
for i, ax in zip(range(0,8,1), axes.flat):
    im = ax.imshow(t_val[:, :, 45, i], cmap = 'RdYlBu')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.savefig("../../../data/maps/full_t_map.png")
plt.close()

############## Visualizing p values for the middle slice in gray
fig, axes = plt.subplots(nrows=2, ncols=4)
for i, ax in zip(range(0,8,1), axes.flat):
    im = ax.imshow(p_val[:, :, 45, i], cmap = 'gray')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.savefig("../../../data/maps/full_p_map.png")
plt.close()


############## significant voxels
# Create contrast matrix for each beta:
t, p = linear_modeling.t_stat(X, [1,1,1,1,1,1,0,0,0], beta, MRSS, df)

## Use Bonferroni correction for setting threshold
threshold = 0.05/n_trs
# the index of the voxels whose p-values are significant
sig_pos = np.where(p <= threshold)
print('The activated voxels under threshold of 0.05/133 are ' + str(sig_pos[1]))
############## plotting of significant voxels
p_val = np.ones(vol_shape + (p.shape[0],))
p_val[in_brain_mask, :] = p.T
linear_modeling.p_map(1,1, p_val[..., 0], threshold)
plt.savefig("../../../data/maps/full_sig_p_map.png")
plt.close()
