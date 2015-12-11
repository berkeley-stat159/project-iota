from __future__ import division
import numpy as np
import pandas as pd
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import t as t_dist
from nilearn import image
from nilearn.plotting import plot_stat_map
from scipy.ndimage import gaussian_filter
import statsmodels.stats.diagnostic

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

    return beta, errors, MRSS, df

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
    t statistics: a vector of length n_vols
        t statistics for each voxel.
    p values: a vector of length n_vols
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

def p_map(f1, p_values_3d):
    """
    Generate three different views of p-map to show the voxels
    where is significantly active

    parameters
    ----------
    data: string contains task#_run#/filtered_func_data_mni
    p_value_3d: 3D array of p_value
    """
    fmri_img = image.smooth_img('../../../data/sub001/BOLD/' + f1 + '/filtered_func_data_mni.nii.gz', fwhm = 6)
    mean_img = image.mean_img(fmri_img)

    log_p_values = -np.log10(p_values_3d)
    log_p_values[np.isnan(log_p_values)] = 0.
    log_p_values[log_p_values > 10.] = 10.
    log_p_values[log_p_values < -np.log10(0.05/133)] = 0
    plot_stat_map(nib.Nifti1Image(log_p_values, fmri_img.get_affine()),
                  mean_img, title="p-values", annotate=False, colorbar=True)
    plt.savefig("../../../data/maps/block_p_map.png")

def smoothing(data, mask):
    """
    Smooth by number of voxel SD in all three spatial dimissions
    
    parameters
    ----------
    data: 4D array of raw data
    smoothing_dim: list of which veoxels are going to smooth
    
    Returns
    ----------
    Y: smoothing raw data
    """
    smooth_data = gaussian_filter(data, [2,2,2,0])
    Y = smooth_data[mask].T

    return Y

def white_test(residual, design):
    residual_square = np.square(residual) #(133, 194287)
    exog = design # (133, 4)
    stat_table = np.zeros((residual_square.shape[1], ))
    for i in range(1, design.shape[1], 1):
        tmp = design[:, 1:].T * design[:, i]
        exog = np.concatenate((exog, tmp.T),1) # (133, 13)

    exog = pd.DataFrame(exog.T).drop_duplicates().values
    exog = exog.T # (133, 10)

    for i in range(residual_square.shape[1]):
        _, stat_table[i], _, _ = statsmodels.stats.diagnostic.het_white(residual_square[:, i], exog, False)

    return stat_table
