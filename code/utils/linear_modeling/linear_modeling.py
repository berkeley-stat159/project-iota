from __future__ import division
import numpy as np
import pandas as pd
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
import numpy.linalg as npl
from scipy.stats import t as t_dist
from nilearn import image
from nilearn.plotting import plot_stat_map
from scipy.ndimage import gaussian_filter
import statsmodels.api as sm
import statsmodels.stats.diagnostic
from sklearn import cross_validation

""" Linear_modeling.py

    We are going to model signal by incorporating five conditions all convolved
with hrf, along with a few linear drift terms.

    This is OLS estimation; we assume the errors to have independent and
identical normal distributions around zero for each i in e_i (i.i.d).
"""

def OLS(design, y):
    """
    parameters
    ----------
    y: 2D array (n_trs x n_vols)
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
    x = np.asarray(design)
    residuals = np.zeros(y.shape)

    # Fit in OLS
    for i in range(y.shape[1]):
        model = sm.OLS(y[:,i],x)
        result = model.fit()
        residuals[:,i] = result.resid

    return residuals

def beta_est(y, X):
    """
    parameters
    ----------
    y: 2D array (n_trs x n_vols)
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
    ltp = t_dist.cdf(t, df)  # lower tail p
    p = 1 - ltp  # upper tail p

    return t, p

def smoothing(data, mask):
    """
    Smoothing by number of voxel SD in all three spatial dimensions

    Parameters
    ----------
    data: 4D array of raw data
    smoothing_dim: list of which veoxels are going to smooth

    Returns
    ----------
    Y: 2D array: n_trs x n_voxels
        raw data to be smoothed
    """
    smooth_data = gaussian_filter(data, [2, 2, 2, 0])

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


def p_map(task, run, p_values_3d, threshold=0.05):
    """
    Generate three thresholded p-value maps.

    Parameters
    ----------
    task: int
        Task number
    run: int
        Run number
    p_value_3d: 3D array of p_value.
    threshold: The cutoff value to determine significant voxels.

    Returns
    -------
    threshold p-value images
    """
    fmri_img = image.smooth_img('../../../data/sub001/BOLD/' + 'task00' +
                                str(task) + '_run00' + str(run) +
                                '/filtered_func_data_mni.nii.gz',
                                fwhm=6)

    mean_img = image.mean_img(fmri_img)

    log_p_values = -np.log10(p_values_3d)
    log_p_values[np.isnan(log_p_values)] = 0.
    log_p_values[log_p_values > 10.] = 10.
    log_p_values[log_p_values < -np.log10(threshold)] = 0
    plot_stat_map(nib.Nifti1Image(log_p_values, fmri_img.get_affine()),
                  mean_img, title="Thresholded p-values",
                  annotate=False, colorbar=True)

def split_data(smooth_data):
    cv = cross_validation.KFold(smooth_data.shape[0], n_folds=5)
    
    return cv