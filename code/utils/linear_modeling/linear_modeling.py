from __future__ import division
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import t as t_dist

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


def reshape(mask, vol_shape, arr):
    """
    We can use 3D mask to index into 4D dataset.
    This selects all the voxel time-courses for voxels within the brain
    (as defined by the mask).

    Parameters
    ----------
    mask: 3D array
        The resulting mask of using mean volumes over time.
    vol_shape: tuple
        3D shape of voxels.
    arr: 2D array (int x n_vols)
        array to be reshaped.
    Returns
    -------
    vol_reshape
    masked voxel time-course

    Intermediate step example for demonstrate:

    To reshape beta:
    b_vols = np.zeros(vol_shape + (beta.shape[0],))
    b_vols[in_brain_mask, :] = beta.T
    """
    vols_reshape = np.zeros(vol_shape + (arr.shape[0],))
    vols_reshape[mask, :] = arr.T

    return vols_reshape


