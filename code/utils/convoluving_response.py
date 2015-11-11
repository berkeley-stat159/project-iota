import numpy as np
import matplotlib.pyplot as plt
import scipy.stats
from scipy.stats import gamma
from stimuli import events2neural
import nibabel as nib

def hrf(times):
    """ Return values for HRF at given times """
    # Gamma pdf for the peak
    peak_values = gamma.pdf(times, 6)
    # Gamma pdf for the undershoot
    undershoot_values = gamma.pdf(times, 12)
    # Combine them
    values = peak_values - 0.35 * undershoot_values
    # Scale max to 0.6
    return values / np.max(values) * 0.6

def get_n_volx(f1):
	img = nib.load('../../data/sub001/BOLD/' + f1 + '.nii.gz')
	data = img.get_data()
	return data.shape[-1]

def constructing_convo(f2, n_volx):
	get_name = f2.replace('/', '_').replace('d', 'v')

	tr_times = np.arange(0, 30, 2.5)
	hrf_at_trs = hrf(tr_times)
	neural_prediction = events2neural('../../data/sub001/model/model001/onsets/' + f2 + '.txt', 2.5, n_volx)
	all_tr_times = np.arange(n_volx) * 2.5
	convolved = np.convolve(neural_prediction, hrf_at_trs)
	convolved = convolved[:(len(convolved) - len(hrf_at_trs) + 1)]
	convolved = np.append(convolved, np.zeros(len(all_tr_times)-len(convolved)))

	plt.plot(all_tr_times, neural_prediction)
	plt.plot(all_tr_times, convolved)
	plt.show()
	plt.savefig('../../data/convo/' + get_name + '.png')
	np.savetxt('../../data/convo/' + get_name + '.txt', convolved)

if __name__ == '__main__':
	from sys import argv

	f1 = argv[1]
	f2 = argv[2]
	n_volx = get_n_volx(f1)
	constructing_convo(f2, n_volx)
