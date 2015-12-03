from __future__ import division
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import matplotlib.colors as pcl
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
plt.hist(np.ravel(mean_data), bins=100)
mask = mean_data > 8000
smooth_data = linear_modeling.smoothing(data, mask)

# Block linear regression
betas_hat, s2, df = linear_modeling.beta_est(smooth_data, design)
np.savetxt('../../../data/beta/' + f2 + '_betas_hat_block.txt', betas_hat.T, newline='\r\n')

# Filling back to raw data shape
beta_vols = np.zeros(vol_shape + (betas_hat.shape[0],))
beta_vols[mask] = betas_hat.T

# set regions outside mask as missing with np.nan
# mean_data[~mask] = np.nan
# beta_vols[~mask] = np.nan

# T-test on null hypothesis, assume only input variance of beta3 [0,0,0,1]
t_value, p_value = linear_modeling.t_stat(design, [0,0,0,1], beta_vols, s2, df)
print(p_value.shape)
# Loading color value for cmap
# cmap_values = np.loadtxt('../../../data/color_map.txt')
# nice_cmap = pcl.ListedColormap(cmap_values, 'actc')
# plt.imshow(mean_data[:, :, 45], cmap='gray', alpha=0.5)
# plt.imshow(beta_vols[:, :, 45, 0], cmap=nice_cmap, alpha=0.5)
# plt.show()

# P value for single voxel
plt.figure(0)
plt.plot(range(p_value.shape[1]), p_value[0,:])
plt.xlabel('voxel')
plt.ylabel('p-value')
line = plt.axhline(0.01, ls='--', color = 'red')
plt.savefig('../../../data/beta/' + f2 + '_p_value_block.png')
np.savetxt('../../../data/beta/' + f2 + '_p-value_block.txt', p_value, newline='\r\n')

# # T value for single voxel
# plt.figure(1)
# plt.plot(range(p_value.shape[1]), t_value[0,:])
# plt.xlabel('voxel')
# plt.ylabel('t-value')
# plt.savefig('../../../data/beta/' + f2 + '_T_value_block.png')
# np.savetxt('../../../data/beta/' + f2 + '_T-value_block.txt', t_value, newline='\r\n')

# # active_voxel = range(p_value.shape[1])[p_value <= 0.015]
# # np.savetxt('../../../data/beta/' + f2 + '_active_voxel.txt', active_voxel, newline='\r\n')

# # generate p-map
# p_values_3d = p_value.reshape(vol_shape)
# linear_modeling.p_map(f1, p_values_3d)
# plt.show()