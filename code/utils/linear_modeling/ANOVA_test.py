from __future__ import division
from sys import argv
import numpy as np
import nibabel as nib
import linear_modeling

"""
set the directory to project-iota and run on terminal:
python ANOVA_test.py task001_run001

"""

############## Load f1, the BOLD image.
f1 = argv[1] #task001_run001
img = nib.load('../../../data/sub001/BOLD/' + f1 + '/filtered_func_data_mni.nii.gz')
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

############## we take the mean volume (over time)
mean_vol = np.mean(data, axis=-1)

# mask out the outer-brain noise using mean volumes over time.
in_brain_mask = mean_vol > 8000
# We can use this 3D mask to index into our 4D dataset.
# This selects all the voxel time-courses for voxels within the brain
# (as defined by the mask)
############## Spatially smoothing the raw data
y = linear_modeling.smoothing(data, in_brain_mask)


############## Lastly, do t test on betas:
X = design_mat

############## Get RSS from full model
_, _, MRSS, df = linear_modeling.beta_est(y,X)
RSS = MRSS * df

############## Test beta1 + beta4 + beta5 = 0 (block design)
index1 = np.array([0, 3, 4])
X_1 = np.delete(X, index1, axis = 1)
_, _, MRSS1, df1 = linear_modeling.beta_est(y,X_1)
RSS1 = MRSS1 * df1	

############## Test beta2 + beta3 = 0 (event related design)
index2 = np.array([1, 2])
X_2 = np.delete(X, index2, axis = 1)
_, _, MRSS2, df2 = linear_modeling.beta_est(y,X_2)
RSS2 = MRSS2 * df2	

############## Compare RSS
print("The RSS of full model is", np.mean(RSS))
print("The RSS of the model without block design is", np.mean(RSS1))
print("The RSS of the model without event related design is", np.mean(RSS2))




