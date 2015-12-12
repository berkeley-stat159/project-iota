from scipy.stats import shapiro
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt
from nilearn.plotting import plot_stat_map
from nilearn import image
import nibable as nib
import linear_modeling
import normal_assumption
import filtering

img = nib.load("../../../data/sub001/BOLD/task003_run001/filtered_func_data_mni.nii.gz")
data = img.get_data()[...,4:]

######### Get n_trs and voxel shape
n_trs = data.shape[-1]
vol_shape = data.shape[:-1]

######### we take the mean volume (over time)
mean_vol = np.mean(data, axis=-1)
# mask out the outer-brain noise using mean volumes over time.
in_brain_mask = mean_vol > 8000
# We can use this 3D mask to index into our 4D dataset.
# This selects all the voxel time-courses for voxels within the brain
# (as defined by the mask)
y = linear_modeling.smoothing(data, in_brain_mask)


#reshape data into 2D
n_vol = np.product(vol_shape)
data = np.reshape(data,(n_trs, n_vol))

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# Block Design using original data
start = np.loadtxt('../../../data/convo/' + f1 + '_conv001.txt')[4:]
end = np.loadtxt('../../../data/convo/' + f1 + '_conv004.txt')[4:]
convo = np.loadtxt('../../../data/convo/' + f1 + '_conv005.txt')[4:]
# Building design X matrix
design = np.ones((len(convo), 4))
design[:, 1] = start
design[:, 2] = end
design[:, 3] = convo
X = design
beta, errors, RSS, df = linear_modeling.beta_est(data, X)

pval = normal_assumption.sw(errors)

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
plot_stat_map(nib.Nifti1Image(log_p_values, img.get_affine()),
              mean_img, title='SW Test p-values for Block Design, original', 
              annotate=False,colorbar=True)
plt.savefig("../../../data/GLS/block_normality_test.png")
#--------------------------------------------------------------------------
# Block Design using smoothed data
start = np.loadtxt('../../../data/convo/' + f1 + '_conv001.txt')[4:]
end = np.loadtxt('../../../data/convo/' + f1 + '_conv004.txt')[4:]
convo = np.loadtxt('../../../data/convo/' + f1 + '_conv005.txt')[4:]
# Building design X matrix
design = np.ones((len(convo), 4))
design[:, 1] = start
design[:, 2] = end
design[:, 3] = convo
X = design
beta, errors, RSS, df = linear_modeling.beta_est(y, X)

pval = normal_assumption.sw(errors)

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
plot_stat_map(nib.Nifti1Image(log_p_values, img.get_affine()),
              mean_img, title='SW Test p-values for Block Design, smoothed', 
              annotate=False,colorbar=True)
plt.savefig("../../../data/GLS/smoothed_block_normality_test.png")




#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
# Full Design using original data
######### Construct design matrix:
# Load the convolution files.
design_mat = np.ones((n_trs, 9))
files = range(1,7,1)
for file in files:
    design_mat[:, (int(file) - 1)] = np.loadtxt('../../../data/convo/task001_run001_conv00'
            + str(file) + '.txt')[4:]
# adding the linear drifter terms to the design matrix
linear_drift = np.linspace(-1, 1, n_trs)
design_mat[:, 6] = linear_drift
quadratic_drift = linear_drift ** 2
quadratic_drift -= np.mean(quadratic_drift)
design_mat[:, 7] = quadratic_drift
X = design_mat

beta, errors, MRSS, df = linear_modeling.beta_est(data,X)

pval = normal_assumption.sw(errors)


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
plot_stat_map(nib.Nifti1Image(log_p_values, img.get_affine()),
              mean_img, title='SW Test p-values for Mixed Design, original', 
              annotate=False, colorbar=True)
plt.savefig("../../../data/GLS/mixed_normality_test.png")
#---------------------------------------------------------------------------
# Full Design using smoothed data

######### Construct design matrix:
# Load the convolution files.
design_mat = np.ones((n_trs, 9))
files = range(1,7,1)
for file in files:
    design_mat[:, (int(file) - 1)] = np.loadtxt('../../../data/convo/task001_run001_conv00'
            + str(file) + '.txt')[4:]
# adding the linear drifter terms to the design matrix
linear_drift = np.linspace(-1, 1, n_trs)
design_mat[:, 6] = linear_drift
quadratic_drift = linear_drift ** 2
quadratic_drift -= np.mean(quadratic_drift)
design_mat[:, 7] = quadratic_drift
X = design_mat

beta, errors, MRSS, df = linear_modeling.beta_est(y,X)

pval = normal_assumption.sw(errors)


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
plot_stat_map(nib.Nifti1Image(log_p_values, img.get_affine()),
              mean_img, title='SW Test p-values for Mixed Design, smoothed', 
              annotate=False, colorbar=True)
plt.savefig("../../../data/GLS/smoothed_mixed_normality_test.png")



#-----------------------------------------------------------------------
#-----------------------------------------------------------------------
# dct Modeling using original data
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
X = dct_design_mat

beta, errors, MRSS, df = linear_modeling.beta_est(data,X)

pval = normal_assumption.sw(errors)


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
plot_stat_map(nib.Nifti1Image(log_p_values, img.get_affine()),
              mean_img, title='SW Test p-values for Mixed Design with dct, original', annotate=False,
              colorbar=True)
plt.savefig("../../../data/GLS/mixed_dct_normality_test.png")

#---------------------------------------------------------------------------
# dct Modeling using smoothed data
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
X = dct_design_mat

beta, errors, MRSS, df = linear_modeling.beta_est(y,X)

pval = normal_assumption.sw(errors)


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
plot_stat_map(nib.Nifti1Image(log_p_values, img.get_affine()),
              mean_img, title='SW Test p-values for Mixed Design with dct, smoothed', annotate=False,
              colorbar=True)
plt.savefig("../../../data/GLS/smoothed_mixed_dct_normality_test.png")

