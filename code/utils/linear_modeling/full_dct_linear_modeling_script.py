from __future__ import division
from sys import argv
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import linear_modeling
import filtering
"""
set the directory to project-iota and run below command on terminal:

python full_dct_linear_modeling_script.py task001_run001
"""
# uncomment next line to run in iPython
#img = nib.load("../../../data/sub001/BOLD/task001_run001/filtered_func_data_mni.nii.gz")

######### Load f1, the BOLD image.
f1 = argv[1] #task001_run001

img = nib.load('../../../data/sub001/BOLD/' + f1 + '/filtered_func_data_mni.nii.gz')
data = img.get_data()
data = data[...,4:]

######### Get n_trs and voxel shape
n_trs = data.shape[-1]
vol_shape = data.shape[:-1]


######### Construct design matrix:
# Load the convolution files.
dct_design_mat = np.ones((n_trs, 15))
files = range(1,7,1)
for file in files:
    dct_design_mat[:, (file - 1)] = np.loadtxt('../../../data/convo/task001_run001_conv00' +
                                               str(file) + '.txt')[4:]

# adding the linear drifter terms to the design matrix
linear_drift = np.linspace(-1, 1, n_trs)
dct_design_mat[:, 6] = linear_drift
quadratic_drift = linear_drift ** 2
quadratic_drift -= np.mean(quadratic_drift)
dct_design_mat[:, 7] = quadratic_drift

## adding the discrete cosine transformation basis to design matrix
N = n_trs
f_mat = np.zeros((N,7))
for i in range(1,7,1):
    for j in range(0,n_trs,1):
        f_mat[j,i] = filtering.dct(i, j, N)

# show the dcf graphically:
dct_design_mat[:, 8:14] = f_mat[...,1:7]
plt.imshow(dct_design_mat[...,8:13], aspect=0.1, cmap='gray', interpolation='nearest')
plt.savefig('../../../data/design_matrix/dct_mat.png')
plt.close()

# show the design matrix graphically:
plt.imshow(dct_design_mat, aspect=0.1, cmap='gray', interpolation='nearest')
plt.savefig('../../../data/design_matrix/dct_design_mat.png')
plt.close()


######### we take the mean volume (over time), and do a histogram of the values
mean_vol = np.mean(data, axis=-1)
# mask out the outer-brain noise using mean volumes over time.
in_brain_mask = mean_vol > 8000
# We can use this 3D mask to index into our 4D dataset.
# This selects all the voxel time-courses for voxels within the brain
# (as defined by the mask)
y = linear_modeling.smoothing(data, in_brain_mask)


######### Lastly, do t test on betas:
X = dct_design_mat

beta, errors, MRSS, df = linear_modeling.beta_est(y,X)
print('The mean MRSS across all voxels using all 6 study conditions is ' + str(np.mean(MRSS)))

# Visualizing betas for the middle slice
# First reshape
b_vols = np.zeros(vol_shape + (beta.shape[0],))
b_vols[in_brain_mask, :] = beta.T
# Then plot them on the same plot with uniform scale
fig, axes = plt.subplots(nrows=2, ncols=3)
for i, ax in zip(range(0,7,1), axes.flat):
    im = ax.imshow(b_vols[:, :, 45, i], cmap = 'gray')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.savefig("../../../data/maps/dct_beta.png")
plt.close()

# To test significance of betas:
# Create contrast matrix for each beta:
c_mat = np.diag(np.array(np.ones((dct_design_mat.shape[1],))))
# t statistics and p values
# Length is the number of voxels after masking
t_mat = np.ones((dct_design_mat.shape[1], y.shape[1]))
p_mat = np.ones((dct_design_mat.shape[1], y.shape[1],))
for i in range(0,dct_design_mat.shape[1], 1):
    t, p = linear_modeling.t_stat(X, c_mat[:,i], beta, MRSS, df)
    t_mat[i,:] = t
    p_mat[i,:] = p

# Reshape t values
t_val = np.zeros(vol_shape + (t_mat.shape[0],))
t_val[in_brain_mask, :] = t_mat.T
# Reshape p values
p_val = np.zeros(vol_shape + (p_mat.shape[0],))
p_val[in_brain_mask, :] = p_mat.T

# Visualizing t values for the middle slice
fig, axes = plt.subplots(nrows=2, ncols=3)
for i, ax in zip(range(0,7,1), axes.flat):
    im = ax.imshow(t_val[:, :, 45, i], cmap = 'RdYlBu')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.savefig("../../../data/maps/dct_t_map.png")
plt.close()

# Visualizing p values for the middle slice in gray
fig, axes = plt.subplots(nrows=2, ncols=3)
for i, ax in zip(range(0,7,1), axes.flat):
    im = ax.imshow(p_val[:, :, 45, i], cmap = 'gray')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.savefig("../../../data/maps/dct_p_map.png")
plt.close()



############## significant voxels
# Create contrast matrix for each beta:
t, p = linear_modeling.t_stat(X, [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0], beta, MRSS, df)
# using Bonferroni correction for setting threshold
threshold = 0.05 / n_trs
sig_pos = np.where(p <= threshold)
print('The activated voxels under threshold of 0.05/133 are ' + str(sig_pos[1]))
# none is significant

############## plotting of significant voxels
# reshape p values
p_val = np.ones(vol_shape + (p.shape[0],))
p_val[in_brain_mask, :] = p.T
########## Beta 1 + ... + Beta 6
linear_modeling.p_map(1 ,1 , p_val[..., 0], threshold)
plt.savefig("../../../data/maps/dct_sig_p_map.png")
plt.close()

# using a higher threshold
threshold = 0.05 / 110
sig_pos = np.where(p <= threshold)
print('The activated voxels under threshold of 0.05/110 are ' + str(sig_pos[1]))
linear_modeling.p_map(1 ,1 , p_val[..., 0], threshold)
plt.savefig("../../../data/maps/dct_sig2_p_map.png")
plt.close()

############## CV select the model
result = np.array([])

cv = linear_modeling.split_data(y)
for traincv, testcv in cv:
    beta, _, _, _ = linear_modeling.beta_est(y[traincv], X[traincv])
    errors = np.subtract(y[testcv], X[testcv].dot(beta))
    result = np.append(result, errors.mean())

print("Mean of squared errors: ", str(result.mean()))