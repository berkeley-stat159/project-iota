from __future__ import division
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
from sys import argv
import linear_modeling

"""
set the directory to project-iota and run below command on terminal:

python linear_modeling_script sub001/BOLD/task001_run001/filtered_func_data_mni

"""

######### set gray colormap and nearest neighbor interpolation by default
#plt.rcParams['image.cmap'] = 'gray'
#plt.rcParams['image.interpolation'] = 'nearest'

######### Load f1, the BOLD image.
#f1 = argv[1] # sub001/BOLD/task001_run001/bold
# or use the filtered data: sub001/BOLD/task001_run001/filtered_func_data_mni

img = nib.load("../../../data/sub001/BOLD/task001_run001/filtered_func_data_mni.nii.gz")
#img = nib.load('../../../data/' + f1 + '.nii.gz')
data = img.get_data()

######### Get n_trs and voxel shape
n_trs = data.shape[-1]
vol_shape = data.shape[:-1]

######### Construct design matrix:
# Load the convolution files.
design_mat = np.ones((n_trs, 9))
files = ('1', '2', '3', '4', '5', '6')
for file in files:
    design_mat[:, (int(file) - 1)] = np.loadtxt('../../../data/convo_prep/task001_run001_cond00'
                                          + file + '_conv.txt')
# adding the linear drifter terms to the design matrix
linear_drift = np.linspace(-1, 1, n_trs)
design_mat[:, 6] = linear_drift
quadratic_drift = linear_drift ** 2
quadratic_drift -= np.mean(quadratic_drift)
design_mat[:, 7] = quadratic_drift
# show the design matrix graphically:
plt.imshow(design_mat, aspect=0.1, cmap='gray')
#plt.show()
#plt.savefig('../../../data/design_matrix/design_mat.png')
#np.savetxt('../../../data/maps/design_mat.txt', design_mat)


######### we take the mean volume (over time), and do a histogram of the values
mean_vol = np.mean(data, axis=-1)
plt.hist(np.ravel(mean_vol), bins=100)
plt.xlabel('Voxels')
plt.ylabel('Frequency')
plt.title('Mean Volume Over Time')
#plt.show()
#plt.savefig("../../../data/design_matrix/mean_vol.png")
# mask out the outer-brain noise using mean volumes over time.
in_brain_mask = mean_vol > 5000
# We can use this 3D mask to index into our 4D dataset.
# This selects all the voxel time-courses for voxels within the brain
# (as defined by the mask)
in_brain_tcs = data[in_brain_mask, :]



######### Lastly, do t test on betas:
y = in_brain_tcs.T
X = design_mat

beta, MRSS, df = linear_modeling.beta_est(y,X)

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
plt.show()

# To test significance of betas:
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
#np.savetxt('../../../data/maps/t_stat.txt', t_mat)
#np.savetxt('../../../data/maps/p_val.txt', p_mat)

# Reshape t values
t_val = np.zeros(vol_shape + (t_mat.shape[0],))
t_val[in_brain_mask, :] = t_mat.T
# Reshape p values
p_val = np.zeros(vol_shape + (p_mat.shape[0],))
p_val[in_brain_mask, :] = p_mat.T

# Visualizing t values for the middle slice
fig, axes = plt.subplots(nrows=2, ncols=4)
for i, ax in zip(range(0,8,1), axes.flat):
    im = ax.imshow(t_val[:, :, 45, i], cmap = 'RdYlBu')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.show()

# Visualizing p values for the middle slice in gray
fig, axes = plt.subplots(nrows=2, ncols=4)
for i, ax in zip(range(0,8,1), axes.flat):
    im = ax.imshow(p_val[:, :, 45, i], cmap = 'gray')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.show()


# significant p values
vols_reshape = np.ones(vol_shape + (9,))
for x in range(0,9,1):
    sig_mask = p_val[...,x] <= (0.05 / y.shape[1])
    arr = np.reshape(p_val[...,x].ravel(), (-1,))
    arr = arr[sig_mask.ravel()]
    vols_reshape[sig_mask, x] = arr.T

plt.imshow(vols_reshape[:,:,45,4])
plt.colorbar()
plt.show()


mod = sm.GLM(np.vstack([yy, 1-yy]).T, sm.add_constant(X), family=sm.families.Binomial()).fit()

