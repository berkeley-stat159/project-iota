from __future__ import division
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import matplotlib.colors
import nibabel as nib
from scipy.stats import t as t_dist
from nilearn import image
from nilearn.plotting import plot_stat_map
import linear_modeling
from sys import argv

"""
set the directory to project-iota and run below command on terminal:
python block_linear_modeling_script.py task001_run001/filtered_func_data_mni task001_run001
"""
f1 = argv[1] # filtered_func_data_mni
f2 = argv[2] # task001_run001

# Load the image as an image object
img = nib.load('../../../data/sub001/BOLD/' + f1 + '.nii.gz')
# Load the image data as an array
# Drop the first 4 3D volumes from the array
data = img.get_data()[..., 4:]
# Load the pre-written convolved time course
start = np.loadtxt('../../../data/convo/' + f2 + '_conv001.txt')[4:]
end = np.loadtxt('../../../data/convo/' + f2 + '_conv004.txt')[4:]
convo = np.loadtxt('../../../data/convo/' + f2 + '_conv005.txt')[4:]

# Building design X matrix
design = np.ones((len(convo), 4))
design[:, 1] = start
design[:, 2] = end
design[:, 3] = convo

# reshape data to 2D
vol_shape, n_time = data.shape[:-1], data.shape[-1]
# shape_2d = (n_time, np.product(vol_shape)) # (133, 902629)

# Smoothing raw data set
mean_data = np.mean(data, -1)
plt.figure(2)
plt.hist(np.ravel(mean_data), bins=100)
line = plt.axvline(8000, ls='--', color = 'red')
mask = mean_data > 8000
smooth_data = linear_modeling.smoothing(data, mask)

# Block linear regression
betas_hat, s2, df = linear_modeling.beta_est(smooth_data, design) #(4, 194287)
np.savetxt('../../../data/beta/' + f2 + '_betas_hat_block.txt', betas_hat.T, newline='\r\n')

# Filling back to raw data shape
beta_vols = np.zeros(vol_shape + (betas_hat.shape[0],)) #(91, 109, 91, 4)
beta_vols[mask] = betas_hat.T

# set regions outside mask as missing with np.nan
mean_data[~mask] = np.nan
beta_vols[~mask] = np.nan

# T-test on null hypothesis, assume only input variance of beta3 [0,0,0,1]
t_value, p_value = linear_modeling.t_stat(design, [0,1,1,1], betas_hat, s2, df) #(1, 194287) (1, 194287)

# Filling T, P value back to raw data shape (91, 109, 91)
t_vols = np.zeros(vol_shape + (t_value.shape[0],))
t_vols[mask, :] = t_value.T
p_vols = np.ones(vol_shape + (p_value.shape[0],))
p_vols[mask, :] = p_value.T

#========================================================================================================
# P value for single voxel
plt.figure(0)
plt.plot(range(p_value.shape[1]), p_value[0,:])
plt.xlabel('voxel')
plt.ylabel('P-value')
plt.title('P-value for single voxel')
line = plt.axhline(0.01, ls='--', color = 'red')
plt.savefig('../../../data/beta/' + f2 + '_p_value_block.png')
np.savetxt('../../../data/beta/' + f2 + '_p-value_block.txt', p_value, newline='\r\n')

# T value for single voxel
plt.figure(1)
plt.plot(range(t_value.shape[1]), t_value[0,:])
plt.xlabel('voxel')
plt.ylabel('t-value')
plt.title('T-value for single voxel')
plt.savefig('../../../data/beta/' + f2 + '_T_value_block.png')
np.savetxt('../../../data/beta/' + f2 + '_T-value_block.txt', t_value, newline='\r\n')

# # Loading color value for cmap
# cmap_values = np.loadtxt('../../../data/color_map.txt')
# nice_cmap = matplotlib.colors.ListedColormap(cmap_values, 'color_map')
# fig, axes = plt.subplots(nrows=4, ncols=4)
# for i, ax in zip(range(25,58,2), axes.flat):
# 	im = ax.imshow(mean_data[:, i, :,], cmap='gray', alpha=0.5)
# 	io = ax.imshow(beta_vols[:, i, :, 3], cmap=nice_cmap, alpha=0.5)
# fig.subplots_adjust(right=0.85)
# cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# fig.suptitle("Middle Level for beta_hat", fontsize=20)
# fig.colorbar(io, cax=cax)

# # Visualizing p values for the middle slice in gray
# fig, axes = plt.subplots(nrows=4, ncols=4)
# for i, ax in zip(range(25,58,2), axes.flat):
#     im = ax.imshow(mean_data[:, i, :,], cmap='gray', alpha=0.5)
#     io = ax.imshow(p_vols[:, i, :, 0], cmap=nice_cmap, alpha=0.5)
# fig.subplots_adjust(right=0.85)
# cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# fig.suptitle("Middle Level for P-value", fontsize=20)
# fig.colorbar(io, cax=cax)

# fig, axes = plt.subplots(nrows=4, ncols=4)
# for i, ax in zip(range(25,58,2), axes.flat):
# 	im = ax.imshow(mean_data[i, :, :,], cmap='gray', alpha=0.5)
# 	io = ax.imshow(beta_vols[i, :, :, 3], cmap=nice_cmap, alpha=0.5)
# fig.subplots_adjust(right=0.85)
# cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# fig.suptitle("Front Level for beta_hat", fontsize=20)
# fig.colorbar(io, cax=cax)

# # Visualizing p values for the middle slice in gray
# fig, axes = plt.subplots(nrows=4, ncols=4)
# for i, ax in zip(range(25,58,2), axes.flat):
#     im = ax.imshow(mean_data[i, :, :,], cmap='gray', alpha=0.5)
#     io = ax.imshow(p_vols[i, :, :, 0], cmap=nice_cmap, alpha=0.5)
# fig.subplots_adjust(right=0.85)
# cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# fig.suptitle("Front Level for P-value", fontsize=20)
# fig.colorbar(io, cax=cax)

# fig, axes = plt.subplots(nrows=4, ncols=4)
# for i, ax in zip(range(25,58,2), axes.flat):
# 	im = ax.imshow(mean_data[:, :, i,], cmap='gray', alpha=0.5)
# 	io = ax.imshow(beta_vols[:, :, i, 3], cmap=nice_cmap, alpha=0.5)
# fig.subplots_adjust(right=0.85)
# cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# fig.suptitle("Back Level for beta_hat", fontsize=20)
# fig.colorbar(io, cax=cax)

# # Visualizing p values for the middle slice in gray
# fig, axes = plt.subplots(nrows=4, ncols=4)
# for i, ax in zip(range(25,58,2), axes.flat):
#     im = ax.imshow(mean_data[:, :, i,], cmap='gray', alpha=0.5)
#     io = ax.imshow(p_vols[:, :, i, 0], cmap=nice_cmap, alpha=0.5)
# fig.subplots_adjust(right=0.85)
# cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# fig.suptitle("Back Level for P-value", fontsize=20)
# fig.colorbar(io, cax=cax)

# # generate p-map
# linear_modeling.p_map(f1, p_vols)
plt.show()

# +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# # Visualizing t values for the middle slice
# fig, axes = plt.subplots(nrows=2, ncols=4)
# for i, ax in zip(range(0,8,1), axes.flat):
#     im = ax.imshow(t_val[:, :, 45, i], cmap = 'RdYlBu')
# fig.subplots_adjust(right=0.85)
# cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
# fig.colorbar(im, cax=cax)
# plt.show()

# # active_voxel = range(p_value.shape[1])[p_value <= 0.015]
# # np.savetxt('../../../data/beta/' + f2 + '_active_voxel.txt', active_voxel, newline='\r\n')
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++