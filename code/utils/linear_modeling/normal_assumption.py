from scipy.stats import shapiro
import scipy.stats as stats
import numpy as np
import matplotlib.pyplot as plt

""" Normal_assumption.py

    In our linear modeling, we assume that errors are normally distributed. To
prove this, we could use the Q-Q plot of errors to check for normality.
"""


def normplot(e):
    """
    parameters
    ----------
    e: error of a single voxel through time

    Returns
    -------
    a Q-Q plot
    """
    stats.probplot(e, dist = "norm", plot = plt)
    plt.title("Normal Q-Q plot")
    plt.show()
    plt.savefig('../../../data/normal_assumption.png')



def sw(errors):
    """
    Shapiro Wilk Test

    The Null hypothesis for SW test is that the data forms a normal 
    distribution.

    Parameters
    -------------
    errors: error of voxels through time (shape of it is 221783*1)

    Returns
    ---------
    swstat: test statistics for SW test
    pval: P-value for the hypothesis test.
    """
    
    pval = []

    for i in range(errors.shape[-1]):
        pval.append(shapiro(errors[:,i])[1])

    pval = np.array(pval)
    shap=pval.shape[0]
    pval = np.reshape(pval, (shap, 1))


    return pval

