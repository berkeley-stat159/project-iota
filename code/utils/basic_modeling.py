import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt
import nibabel as nib

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
    data_2d = np.reshape(data, (-1, data.shape[-1]))
    betas = npl.pinv(design).dot(data_2d.T) #(2, 147456)
    betas_4d = np.reshape(betas.T, data.shape[:-1] + (-1,)) #(64, 64, 36, 2)

    return (betas, betas_4d)

if __name__ == '__main__':
    from sys import argv
    f1 = argv[1]
    f2 = argv[2]
    get_name = f2.replace('/', '_')
    data, convolved = load_data(f1, f2)
    betas_hat, betas_hat_4d = reg_voxels_4d(data, convolved)
    np.savetxt('../../data/beta/' + get_name + '.txt', betas_hat, newline='\r\n')
    plt.imshow(betas_hat_4d[:, :, 14, 0])
    plt.savefig('../../data/beta/task001_run001_conv005.png')
    plt.show()

