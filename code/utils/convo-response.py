from __future__ import division
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
	img = nib.load('../../data/' + f1 + '.nii.gz')
	data = img.get_data()
	return data.shape[-1]

def rescale_cond(f2, n_volx):
	cond_data = np.loadtxt('../../data/sub001/model/model001/onsets/' + f2 + '.txt')
	getname = f2.replace('/', '_')
	# read into three components
	onsets = cond_data[:, 0]
	duration = cond_data[:, 1]
	ampl = cond_data[:, 2]
	# finer resolution has 100 steps per TR
	tr_divs = 100.0
	TR = 2.5
	high_res_times = np.arange(0, n_volx, 1/tr_divs) * TR
	# create a new neural prediction time-course for 1/100 of TR
	high_res_neural = np.zeros(high_res_times.shape)
	high_res_onset_indices = onsets / TR * tr_divs
	high_res_duration = duration / TR * tr_divs
	# fill in the high_res_neural time course
	for hr_onset, hr_duration, amplitude in zip(high_res_onset_indices, high_res_duration, ampl):
		hr_onset = int(round(hr_onset))
		hr_duration = int(round(hr_duration))
		high_res_neural[hr_onset:hr_onset + hr_duration] = amplitude
	
	plt.plot(high_res_times, high_res_neural)
	plt.xlabel('Time (Seconds)')
	plt.ylabel('High resolution neural prediction')
	plt.savefig('../../data/convo/High_resolution_neural_' + getname + '.png')

	return (high_res_neural, high_res_times, getname)


def constructing_convo(high_res_neural, high_res_times, getname):
	tr_divs = 100
	hrf_times = np.arange(0, 24, 1/tr_divs)
	hrf_at_trs = hrf(hrf_times)
	
	high_res_hemo = np.convolve(high_res_neural, hrf_at_trs)[:len(high_res_neural)]
	plt.plot(high_res_times, high_res_hemo)
	plt.xlabel('Time (Seconds)')
	plt.ylabel('High resolution convolved values')
	plt.savefig('../../data/convo/High_resolution_convo_' + getname + '.png')

	# neural_prediction = events2neural('../../data/sub001/model/model001/onsets/' + f2 + '.txt', 2.5, n_volx)
	# all_tr_times = np.arange(n_volx) * 2.5
	# convolved = np.convolve(neural_prediction, hrf_at_trs)
	# convolved = convolved[:(len(convolved) - len(hrf_at_trs) + 1)]
	# convolved = np.append(convolved, np.zeros(len(all_tr_times)-len(convolved)))

	# plt.plot(all_tr_times, neural_prediction)
	# plt.plot(all_tr_times, convolved)
	# plt.show()
	# plt.savefig('../../data/convo/' + get_name + '.png')
	# np.savetxt('../../data/convo/' + get_name + '.txt', convolved)

if __name__ == '__main__':
	from sys import argv

	f1 = argv[1] #task
	f2 = argv[2]
	n_volx = get_n_volx(f1)
	high_res_neural, high_res_times, getname = rescale_cond(f2, n_volx)
	constructing_convo(high_res_neural, high_res_times, getname)

