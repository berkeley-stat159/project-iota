from scipy.stats import shapiro
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn import image
import nibable as nib
import linear_modeling
import normal_assumption

img = nib.load("../../../data/sub001/BOLD/task003_run001/filtered_func_data_mni.nii.gz")
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

######### we take the mean volume (over time), and do a histogram of the values
mean_vol1 = np.mean(data, axis=-1)

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

errors = y - X.dot(beta)

pval = sw(errors)

#The return, pval should have shape (221783,1) which is a column vector.
# Reshape p values

p_val = np.ones(vol_shape + (pval.shape[1],))
p_val[in_brain_mask, :] = pval



# smoothing
fmri_img = image.smooth_img('../../../data/sub001/BOLD/task001_run001/filtered_func_data_mni.nii.gz', fwhm=6)
mean_img = image.mean_img(fmri_img)
# Thresholding
p_val = np.ones(vol_shape + (pval.shape[1],))
p_val[in_brain_mask, :] = pval

log_p_values = -np.log10(p_val[..., 0])
log_p_values[np.isnan(log_p_values)] = 0.
log_p_values[log_p_values > 10.] = 10.
log_p_values[log_p_values < -np.log10(0.05/137)] = 0
plot_stat_map(nibabel.Nifti1Image(log_p_values, img.get_affine()),
              mean_img, title='SW Test p-values', annotate=False,
              colorbar=True)
