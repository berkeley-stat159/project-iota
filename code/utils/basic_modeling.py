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
    design = np.ones((len(convolved), 2))
    design[:, 1] = convolved
    data_2d = np.reshape(data, (data.shape[-1], -1))
    betas = npl.pinv(design).dot(data_2d) #(2, 147456)
    betas_4d = np.reshape(betas.T, data.shape[:-1] + (-1,)) #(64, 64, 36, 2)
    print(design.shape, data_2d.shape, betas.shape)
    return (design, data_2d, betas, betas_4d)   

def RSE(X,Y, betas_hat): #input 2D of X, Y and betas_hat
    Y_hat = X.dot(betas_hat)
    res = Y - Y_hat
    RSS = np.sum(res ** 2, 0)
    print(RSS.shape)
    df = X.shape[0] - npl.matrix_rank(X)
    MRSS = RSS / df

    return (MRSS, df) 

def hypothesis(betas_hat, v_cov, df):
    t_stat = np.array([])
    for i in v_cov:
        t_stat = np.append(t_stat, betas_hat[:, 1] / np.sqrt(i[1,1]))

    # Get p value for t value suing cumulative density dunction
    p = np.array([])
    for j in t_stat:
        p = np.append(p, 1 - t_dist.cdf(j, df))

    return p

if __name__ == '__main__':
    from sys import argv
    f1 = argv[1] #task001_run001/bold
    f2 = argv[2] #task001_run001_conv005
    get_name = f2.replace('/', '_')
    data, convolved = load_data(f1, f2)
    design_mat, data_2d, betas_hat, betas_hat_4d = reg_voxels_4d(data, convolved)
    np.savetxt('../../data/beta/' + get_name + '.txt', betas_hat, newline='\r\n')
    plt.imshow(betas_hat_4d[:, :, 15, 1])
    plt.savefig('../../data/beta/task001_run001_conv005.png')
    #plt.show()
    #get residual standard error
    s2, df = RSE(design_mat, data_2d, betas_hat)
    
    #get estimator of variance of betas_hat   
    beta_cov = np.array([])
    for i in s2:
        beta_cov = np.append(beta_cov, npl.inv(design_mat.T.dot(design_mat)))
    beta_cov = beta_cov.reshape(-1,2,2)

    #T-test on null hypothesis
    p_value = hypothesis(betas_hat, beta_cov, df)
    print(p_value)
    np.savetxt(get_name + '_p-value.txt', p_value, newline='\r\n')
