import numpy as np
import scipy.stats
from scipy.stats import gamma
from stimuli import events2neural

def hrf(times):
    peak_values = gamma.pdf(times,6)
    undershoot_values = gamma.pdf(times, 12)
    values = peak_values-0.35*undershoot_values
    
    return values/np.max(values)*0.6

TR = 2.5
tr_times = np.arange(0,30,TR)
hrf_at_trs = hrf(tr_times)

n_vols = 173
neural_prediction = events2neural('cond_filename.txt',TR, n_vols)
all_tr_times = np.arange(173) * TR
convolved = np.convolve(neural_prediction, hrf_at_trs)
N = len(neural_prediction)
M = len(hrf_at_trs)

n_to_remove = len(hrf_at_trs) - 1
convolved = convolved[:-n_to_remove]
np.savetxt('conv_filename.txt', convolved)




