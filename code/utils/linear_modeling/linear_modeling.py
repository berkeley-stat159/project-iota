from __future__ import division
import matplotlib.pyplot as plt
import numpy as np
import nibabel as nib
import numpy.linalg as npl
from scipy.stats import t as t_dist
from nilearn import image
from nilearn.plotting import plot_stat_map
from scipy.ndimage import gaussian_filter

""" Linear_modeling.py

    We are going to model signal by incorporating five conditions all convolved
with hrf, along with a few linear drift terms.

    This is OLS estimation; we assume the errors to have independent and
identical normal distributions around zero for each i in e_i (i.i.d).
"""

def beta_est(y, X):
    """
    parameters
    ----------
    y: 2D array (n_vols x n_trs)
        BOLD data.
    X: 2D array (n_trs * number of regressors)
        design matrix.

    Returns
    ______
    betas: 2D array (number of regressors x n_vols)
        estimated betas for linear model.
    MRSS: 1D array of length n_volx
        Mean residual sum of squares.
    df: int
        n - rank of X.
    """
    # Make sure y, X are all arrays
    y = np.asarray(y)
    X = np.asarray(X)
    # Calculate the parameters - b hat
    beta = npl.pinv(X).dot(y)
    # The fitted values - y hat
    fitted = X.dot(beta)
    # Residual error
    errors = y - fitted
    # Residual sum of squares
    RSS = (errors**2).sum(axis=0)
    # Degrees of freedom is the number of observations n minus the number
    # of independent regressors we have used.  If all the regressor
    # columns in X are independent then the (matrix rank of X) == p
    # (where p the number of columns in X). If there is one column that
    # can be expressed as a linear sum of the other columns then
    # (matrix rank of X) will be p - 1 - and so on.
    df = X.shape[0] - npl.matrix_rank(X)
    # Mean residual sum of squares
    MRSS = RSS / df

    return beta, MRSS, df

def t_stat(X, c, beta, MRSS, df):
    """
    parameters
    ----------
    X: 2D array (n_trs * number of regressors)
        design matrix.
    c: a contrast vector.
    betas: 2D array (number of regressors x n_vols)
        estimated betas for linear model.
    MRSS: 1D array of length n_volx
        Mean residual sum of squares.
    df: int
        n - rank of X.

    Returns
    ______
    t: a vector of length n_vols
        t statistics for each voxel.
    p: a vector of length n_vols
        p values for each voxel.
    """
    X = np.asarray(X)
    c = np.atleast_2d(c).T
    # calculate bottom half of t statistic
    SE = np.sqrt(MRSS * c.T.dot(npl.pinv(X.T.dot(X)).dot(c)))
    t = c.T.dot(beta) / SE
    # Get p value for t value using cumulative density dunction
    # (CDF) of t distribution
    ltp = t_dist.cdf(t, df) # lower tail p
    p = 1 - ltp # upper tail p

    return t, p

def p_map(task, run, p_values_3d, threshold = 0.05):
    """
    Generate three thresholded p-value maps.

    Parameters
    ----------
    task: a string indicates task#
    run: a string indicates run#
    /filtered_func_data_mni.
    p_value_3d: 3D array of p_value.
    threshold: The cutoff value to determine significant voxels.

    Returns
    -------
    threshold p-value images
    """
    fmri_img = image.smooth_img('../../../data/sub001/BOLD/' + task + run +
    '/filtered_func_data_mni.nii.gz', fwhm = 6)

    mean_img = image.mean_img(fmri_img)

    log_p_values = -np.log10(p_values_3d)
    log_p_values[np.isnan(log_p_values)] = 0.
    log_p_values[log_p_values > 10.] = 10.
    log_p_values[log_p_values < -np.log10(threshold)] = 0
    plot_stat_map(nib.Nifti1Image(log_p_values, fmri_img.get_affine()),
                  mean_img, title="p-values", annotate=False, colorbar=True)

def smoothing(data, mask):
    """
    Smoothing by number of voxel SD in all three spatial dimensions

    Parameters
    ----------
    data: 4D array of raw data
    smoothing_dim: list of which veoxels are going to smooth

    Returns
    ----------
    Y: raw data to be smoothed
    """
    smooth_data = gaussian_filter(data, [2,2,2,0])
    Y = smooth_data[mask].T

    return Y


