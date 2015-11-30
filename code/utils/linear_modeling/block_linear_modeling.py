from __future__ import division
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import t as t_dist

def reg_voxels_4d(data, convo, start, end):
    #Create design X matrix with dim (133, 4)
    design = np.ones((len(convo), 4))
    design[:, 1] = start
    design[:, 2] = end
    design[:, 3] = convo
    #Reshape data to dim (133, 902629)
    data_2d = np.reshape(data, (data.shape[-1], -1))
    #Computing betas based on linear modeling with dim (4, 902629)
    betas = npl.pinv(design).dot(data_2d)
    #Reshape betas to betas_4d 
    betas_4d = np.reshape(betas.T, data.shape[:-1] + (-1,)) # (91, 109, 91, 4)
    
    return (design, data_2d, betas, betas_4d)

def RSE(X,Y, betas_hat): #input 2D of X, Y and betas_hat
    #Computing fitted value, residual, RSS, and return RSE
    Y_hat = X.dot(betas_hat)
    res = Y - Y_hat
    RSS = np.sum(res ** 2, 0)
    df = X.shape[0] - npl.matrix_rank(X)
    MRSS = RSS / df
    
    return (MRSS, df) 

def hypothesis(betas_hat, v_cov, df): #assume only input variance of beta3
    # Get std of beta3_hat
    sd_beta = np.sqrt(v_cov)
    # Get T-value
    t_stat = betas_hat[3, :] / sd_beta
    # Get p value for t value using cumulative density dunction
    ltp = np.array([])
    for i in t_stat:
        ltp = np.append(ltp, t_dist.cdf(i, df)) #lower tail p
    p = 1 - ltp

    return p, t_stat