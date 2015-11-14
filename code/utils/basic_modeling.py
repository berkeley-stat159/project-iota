from __future__ import division
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib
from scipy.stats import t as t_dist

def load_data(f1, f2):
    """
    Return the image data as an array and the convolved time course

    Parameters
    ----------
    f1,f2 : string
        The name of data to process
    Returns
    -------
    tuple:  
        Contains the image data as an array and the convolved time course
    """
    # Load the image as an image object
    img = nib.load('../../data/sub001/BOLD/' + f1 + '.nii.gz')
    # Load the image data as an array
    # Drop the first 4 3D volumes from the array
    data = img.get_data()[..., 4:]
    # Load the pre-written convolved time course
    convolved = np.loadtxt('../../data/convo/' + f2 + '.txt')[4:]
    return(data, convolved)

def reg_voxels_4d(data, convolved):
    #Create design X matrix with dim (133, 2)
    design = np.ones((len(convolved), 2))
    design[:, 1] = convolved
    #Reshape data to dim (133, 147456)
    data_2d = np.reshape(data, (data.shape[-1], -1))
    #Computing betas based on linear modeling with dim (2, 147456)
    betas = npl.pinv(design).dot(data_2d)
    #Reshape betas to betas_4d
    betas_4d = np.reshape(betas.T, data.shape[:-1] + (-1,)) #(64, 64, 36, 2)
    
    return (design, data_2d, betas, betas_4d)   

def RSE(X,Y, betas_hat): #input 2D of X, Y and betas_hat
    #Computing fitted value, residual, RSS, and return RSE
    Y_hat = X.dot(betas_hat)
    res = Y - Y_hat
    RSS = np.sum(res ** 2, 0)
    df = X.shape[0] - npl.matrix_rank(X)
    MRSS = RSS / df
    
    return (MRSS, df) 

def hypothesis(betas_hat, v_cov, df): #assume only input variance of beta1
    # Get std of beta1_hat
    sd_beta = np.sqrt(v_cov)
    # Get T-value
    t_stat = betas_hat[1, :] / sd_beta
    # Get p value for t value using cumulative density dunction
    ltp = np.array([])
    for i in t_stat:
        ltp = np.append(ltp, t_dist.cdf(i, df)) #lower tail p
    p = 1 - ltp

    return p, t_stat

if __name__ == '__main__':
    from sys import argv
    f1 = argv[1] #task001_run001/bold_mcf_brain
    f2 = argv[2] #task001_run001_conv005
    get_name = f2.replace('/', '_')
    data, convolved = load_data(f1, f2)
    design_mat, data_2d, betas_hat, betas_hat_4d = reg_voxels_4d(data, convolved)
    np.savetxt('../../data/beta/' + get_name + '.txt', betas_hat.T, newline='\r\n')
    #plt.imshow(betas_hat_4d[:, :, 15, 1])
    #plt.savefig('../../data/beta/task001_run001_conv005.png')
    
    #get residual standard error
    s2, df = RSE(design_mat, data_2d, betas_hat)
   
    #get estimator of variance of betas_hat   
    beta_cov = np.array([])
    for i in s2:
        beta_cov = np.append(beta_cov, i*npl.inv(design_mat.T.dot(design_mat))[1,1])
   
    #T-test on null hypothesis
    p_value, t_value = hypothesis(betas_hat, beta_cov, df)
    plt.figure(0)
    plt.plot(range(data_2d.shape[1]), p_value)
    plt.xlabel('volx')
    plt.ylabel('P-value')
    line = plt.axhline(0.1, ls='--', color = 'red')
    plt.savefig('p_value.png')
    np.savetxt(get_name + '_p-value.txt', p_value, newline='\r\n')

    plt.figure(1)
    plt.plot(range(data_2d.shape[1]), t_value)
    plt.xlabel('volx')
    plt.ylabel('T-value')
    plt.savefig('T_value.png')
    np.savetxt(get_name + '_T-value.txt', t_value, newline='\r\n')


