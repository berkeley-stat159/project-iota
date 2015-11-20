import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from scipy.stats import norm
from sklearn.neighbors import KernelDensity
from sklearn.grid_search import GridSearchCV

def load_data(f1):
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

    return data

def smoothing(data, width):
	kde = KernelDensity(kernel='gaussian', bandwidth=width).fit(data)

	return kde.score_samples(data)

def best_bandwidth(data):
	grid = GridSearchCV(KernelDensity(), {'bandwidth': np.linspace(0.1, 1.0, 30)}, cv = 10)
	grid.fit(data[:, None])

	return grid.best_params_['bandwidth']

if __name__ == '__main__':
	from sys import argv

	filename = argv[1]
	data = load_data('../../data/sub001/BOLD/' + filename)
	data_2d = data.reshape(-1, data.shape[-1])
	
	simulation = np.ones(data_2d.shape)
	for i in range(data_2d.shape[1]):
		best_width = best_bandwidth(data_2d[:,i])
		simulation[:,i] = smoothing(data_2d[:,i], best_width)
	np.savetxt("simulated_data.txt", simulation, , newline='\r\n')
