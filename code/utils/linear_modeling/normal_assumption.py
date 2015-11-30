import scipy.stats as stats
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
from sys import argv


""" Normal_assumption.py

    In our linear modeling, we assume that errors are normally distributed. To
prove this, we could use the Q-Q plot of errors to check for normality.
"""


######### Load f1, the BOLD image.
f1 = argv[1] # sub001/BOLD/task001_run001/bold
# or use the filtered data: sub001/BOLD/task001_run001/filtered_func_data_mni

#img = nib.load("../../../data/sub001/BOLD/task001_run001/filtered_func_data_mni.nii.gz")
img = nib.load('../../../data/' + f1 + '.nii.gz')
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
mean_vol = np.mean(data, axis=-1)
# mask out the outer-brain noise using mean volumes over time.
in_brain_mask = mean_vol > 5000
# We can use this 3D mask to index into our 4D dataset.
# This selects all the voxel time-courses for voxels within the brain
# (as defined by the mask)
in_brain_tcs = data[in_brain_mask, :]


y = in_brain_tcs.T
X = design_mat

# Calculate the parameters - b hat
beta = npl.pinv(X).dot(y)
# The fitted values - y hat
fitted = X.dot(beta)
# Residual error
errors = y - fitted

def normplot(e):
    """
    parameters
    ----------
    e: error of a single voxel through time

    Returns
    -------
    a Q-Q plot
    """
    stats.ptobplot(e, dist = "norm", plot = plt)
    plt.title("Normal Q-Q plot")
    plt.show()
    plt.savefig('../../../data/normal_assumption.png')
