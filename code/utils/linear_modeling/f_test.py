from __future__ import division
import numpy as np
import numpy.linalg as npl
from scipy.stats import f as f_dist



def f_test(y, X, betas):
	"""
	parameters
	----------
	y: 2D array (n_trs x n_vols)
	    BOLD data.
	X: 2D array (n_trs * number of regressors)
	    design matrix.
	    betas: 2D array (number of regressors x n_vols)
	    estimated betas for linear model.

	Returns
	______
	f_value: a vector of length n_vols
		f statistics for each voxel.
	p value: a vector of length n_vols
		p values for each voxel.
	"""

	# Make sure y, X are all arrays
	y = np.asarray(y)
	X = np.asarray(X)

	# The null hypothesis H0 says that the explanatory variable Xj can 
	# be dropped from the linear model.

	# For the full model: M
	# The fitted values - y hat
	fitted = X.dot(betas)
	# Residuals
	errors = y - fitted
	# Residual sum of squares
	RSS_M = (errors**2).sum(axis=0)
	# Degrees of freedom is the number of observations minus the number
	# of independent regressors we have used.  If all the regressor
	# columns in X are independent then the rank(X) = p
	# (where p the number of columns in X). If there is one column that
	# can be expressed as a linear sum of the other columns then
	# rank(X)=p - 1, and so on.
	df = X.shape[0] - npl.matrix_rank(X)
	# Mean residual sum of squares
	MRSS = RSS_M / df

	# For the reduced model: m
	# Variable Xj is dropped here.
	f_value = []
	p_value = []
	for j in range(X.shape[1]):
		X_j = np.delete(X, j, axis = 1)
		beta_j = npl.pinv(X_j).dot(y)
		fitted_j = X_j.dot(beta_j)
		errors_j = y - fitted_j
		RSS_m = (errors_j ** 2).sum(axis = 0)
		fst = (RSS_m - RSS_M) / MRSS
		f_value.append(fst)
		p_value.append(1 - f_dist.cdf(fst, 1, df))

	return f_value, p_value


