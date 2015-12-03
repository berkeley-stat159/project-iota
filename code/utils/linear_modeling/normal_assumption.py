import scipy.stats as stats
import numpy as np
import numpy.linalg as npl
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
