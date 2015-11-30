from __future__ import division
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import t as t_dist
import block_linear_modeling

# input file name
f1 = 'filtered_func_data_mni'
f2 = 'task001_run001'

# Load the image as an image object
img = nib.load('../../../data/sub001/BOLD/' + f1 + '.nii.gz')

# Load the image data as an array
# Drop the first 4 3D volumes from the array
data = img.get_data()[..., 4:]

# Load the pre-written convolved time course
start = np.loadtxt('../../../data/convo/' + f2 + '_cond001.txt')[4:]
end = np.loadtxt('../../../data/convo/' + f2 + '_cond004.txt')[4:]
convo = np.loadtxt('../../../data/convo/' + f2 + '_cond005.txt')[4:]

# Calulate linear regression
design_mat, data_2d, betas_hat, betas_hat_4d = reg_voxels_4d(data, convo, start, end)
np.savetxt('../../data/beta/' + f2 + '_betas_hat_block.txt', betas_hat.T, newline='\r\n')

#get residual standard error
s2, df = RSE(design_mat, data_2d, betas_hat)

#get estimator of variance of betas_hat 
beta_cov = np.array([])
for i in s2:
    beta_cov = np.append(beta_cov, i*npl.inv(design_mat.T.dot(design_mat))[3,3])

# T-test on null hypothesis
p_value, t_value = hypothesis(betas_hat, beta_cov, df)

# P value for single voxel
plt.figure(0)
plt.plot(range(data_2d.shape[1]), p_value)
plt.xlabel('volx')
plt.ylabel('P-value')
line = plt.axhline(0.02, ls='--', color = 'red')
plt.savefig('../../data/beta/p_value_block.png')
np.savetxt('../../data/beta/'+ f2 + '_p-value_block.txt', p_value, newline='\r\n')

# T value for single voxel
plt.figure(1)
plt.plot(range(data_2d.shape[1]), t_value)
plt.xlabel('volx')
plt.ylabel('T-value')
plt.savefig('../../data/beta/T_value_block.png')
np.savetxt('../../data/beta/' + f2 + '_T-value_block.txt', t_value, newline='\r\n')