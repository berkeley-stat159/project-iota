from __future__ import division
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
import linear_modeling
from scipy import signal
from scipy.fftpack import fft, fftshift
import filtering
from nilearn import image
from nilearn.plotting import plot_stat_map

img = nib.load("../../../data/sub001/BOLD/task001_run001/filtered_func_data_mni.nii.gz")
data = img.get_data()
data = data[..., 4:]
n_trs = data.shape[-1]
vol_shape = data.shape[:-1]
######### Construct design matrix:
# Load the convolution files.
design_mat_filter = np.ones((n_trs, 15))
files = ('1', '2', '3', '4', '5', '6')
for file in files:
    design_mat_filter[:, (int(file) - 1)] = np.loadtxt('../../../data/convo_prep/task001_run001_cond00'
                                          + file + '_conv.txt')
# adding the linear drifter terms to the design matrix
linear_drift = np.linspace(-1, 1, n_trs)
design_mat_filter[:, 6] = linear_drift
quadratic_drift = linear_drift ** 2
quadratic_drift -= np.mean(quadratic_drift)
design_mat_filter[:, 7] = quadratic_drift
## adding the discrete cosine transformation basis to design matrix
N = n_trs
f_mat = np.zeros((N,7))
for i in range(1,7,1):
    for j in range(0,n_trs,1):
        f_mat[j,i] = dcf(i, j, N)
design_mat_filter[:, 8:14] = f_mat[...,1:7]
plt.imshow(design_mat_filter[...,8:13], aspect=0.1, cmap='gray', interpolation='nearest')
plt.savefig('../../../data/design_matrix/design_mat_filter.png')
# show the design matrix graphically:
plt.imshow(design_mat_filter, aspect=0.1, cmap='gray', interpolation='nearest')
plt.show()
plt.savefig('../../../data/design_matrix/design_mat_filter.png')
np.savetxt('../../../data/maps/design_mat_filter.txt', design_mat_filter)



#################### temporal filtering
#### Subtract Gaussian weighted running line fit to smooth
mean_vol = np.mean(data, axis=-1)
in_brain_mask = mean_vol > 5000

y = np.loadtxt('../../../data/maps/y.txt')
y_mat = np.zeros(y.shape)
for i in range(y.shape[1]):
    y_mat[:,i] = y[:,i] - gaussian_smooth(y[:,i], 15, 8)[:n_trs]
X = design_mat_filter

beta, MRSS, df = linear_modeling.beta_est(y_mat,X)

b_vols = np.zeros(vol_shape + (beta.shape[0],))
b_vols[in_brain_mask, :] = beta.T
# Then plot them on the same plot with uniform scale
fig, axes = plt.subplots(nrows=3, ncols=4)
for i, ax in zip(range(0,design_mat_filter.shape[1],1), axes.flat):
    im = ax.imshow(b_vols[:, :, 45, i], cmap = 'gray')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.show()


c_mat = np.diag(np.array(np.ones((design_mat_filter.shape[1],))))
# t statistics and p values
# Length is the number of voxels after masking
t_mat = np.ones((design_mat_filter.shape[1], y.shape[1]))
p_mat = np.ones((design_mat_filter.shape[1], y.shape[1],))
for i in range(0,design_mat_filter.shape[1], 1):
    t, p = linear_modeling.t_stat(X, c_mat[:,i], beta, MRSS, df)
    t_mat[i,:] = t
    p_mat[i,:] = p
# save the t values and p values in txt files.

t_val = np.zeros(vol_shape + (t_mat.shape[0],))
t_val[in_brain_mask, :] = t_mat.T
# Reshape p values
p_val = np.zeros(vol_shape + (p_mat.shape[0],))
p_val[in_brain_mask, :] = p_mat.T

fig, axes = plt.subplots(nrows=4, ncols=4)
for i, ax in zip(range(0,design_mat_filter.shape[1] - 1, 1), axes.flat):
    im = ax.imshow(t_val[:, :, 45, i], cmap = 'RdYlBu')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.show()

# Visualizing p values for the middle slice in gray
fig, axes = plt.subplots(nrows=3, ncols=4)
for i, ax in zip(range(0,design_mat_filter.shape[1] - 1,1), axes.flat):
    im = ax.imshow(p_val[:, :, 45, i], cmap = 'gray')
fig.subplots_adjust(right=0.85)
cax = fig.add_axes([0.9, 0.15, 0.03, 0.7])
fig.colorbar(im, cax=cax)
plt.show()



fmri_img = image.smooth_img('../../../data/sub001/BOLD/task001_run001/filtered_func_data_mni.nii.gz', fwhm=6)
mean_img = image.mean_img(fmri_img)
# Thresholding
p_val = np.ones(vol_shape + (p_mat.shape[0],))
p_val[in_brain_mask, :] = p_mat.T

log_p_values = -np.log10(p_val[..., 6])
log_p_values[np.isnan(log_p_values)] = 0.
log_p_values[log_p_values > 10.] = 10.
log_p_values[log_p_values < -np.log10(0.05/137)] = 0
plot_stat_map(nib.Nifti1Image(log_p_values, img.get_affine()),
              mean_img, title='Thresholded p-values after Filtering (beta 6)', annotate=False,
              colorbar=True)

################### Noise from our model
# voxel time course
v_ind = 10000
fitted = X.dot(beta)
resid = y_mat[:,v_ind] - fitted[:,v_ind]
# voxel time course
plt.plot(np.arange(n_trs), resid)
plt.xlabel('TRs')
plt.title('The 10000th Voxel Time Course')
plt.show()
plt.savefig('../../../data/maps/timecourse_vol.png')

########## Noise has structure
ps = np.abs(np.fft.fft(resid))**2
time_step = 1 / n_trs
freqs = np.fft.fftfreq(resid.size, time_step)
idx = np.argsort(freqs)
plt.plot(freqs[idx], ps[idx])
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('The 10000th Voxel Power Spectral Density')
plt.show()
plt.savefig('../../../data/maps/power_spectrum_vol.png')
#########

















# solution:
########## SPM to remove low frequency noise
########## add a discrete cosine transform basis set to design matrix
N =n_trs
f_mat = np.zeros((N,7))
for i in range(1,7,1):
    for j in range(0,n_trs,1):
        f_mat[j,i] = dcf(i, j, N)
##### Gaussian smoothing
#### First fit a Gaussian Weighted running line
#### Fit at time t is a weighted average of data around t.
plot(resid)
plot(gaussian_smooth(resid, 15, 8)[:n_trs])
plt.xlabel('TRs')
plt.ylabel('BOLD')
gaus_smooth = gaussian_smooth(resid, 15, 8)[:n_trs]
plt.plot(resid - gaus_smooth)
np.savetxt('../../../data/maps/gaus_smooth.txt')

#### Then subtract Gaussian weighted running line fit to smooth
y_mat = np.zeros(y.shape)
for i in range(y.shape[1]):
    y_mat[:,i] = y[:,i] - gaussian_smooth(y[:,i], 15, 8)[:n_trs]

ps = np.abs(np.fft.fft(aa))**2
time_step = 1 / n_trs
freqs = np.fft.fftfreq(aa.size, time_step)
idx = np.argsort(freqs)
plt.plot(freqs[idx[64:]], ps[idx[64:]])
plt.xlabel('Frequency')
plt.ylabel('Power')
plt.title('The 10000th Voxel Power Spectral Density')
plt.show()


