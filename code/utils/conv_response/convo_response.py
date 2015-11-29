from __future__ import division
import numpy as np
from scipy.stats import gamma
""" conv-response.py

Given that our data condition files contains trials that are not evenly spaced,
we created a collection of utility functions for high resolution convolution.

See test_* functions in this directory for nose tests.
"""


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


def rescale_cond(cond_data, n_trs, TR, tr_divs):
    """
    Parameters
    ----------
    cond_data: txt
        Condition files that may require finer time resolution.
    n_trs: int
        Total number of TRs.
    TR: int
        The specified TR for scanning.
    tr_div: int
        The level of time resolution desired.

    Returns
    -------
    high_res_times: array
        One dimensional array where one element corresponds to time
        intervals of 1/100 of a TR.
    high_res_neural: array
        one dimensional array of the new neural prediction time-course where one
        element corresponds to 1 / 100 of a TR.
    """
    """ read into three components"""
    onsets = cond_data[:, 0]
    duration = cond_data[:, 1]
    ampl = cond_data[:, 2]

    """ finer resolution has 100 steps per TR"""
    high_res_times = np.arange(0, n_trs, 1/tr_divs)*TR

    """ create a new neural prediction time-course for 1/100 of TR"""
    high_res_neural=np.zeros(high_res_times.shape)
    high_res_onset_indices = onsets / TR * tr_divs
    high_res_duration = duration / TR * tr_divs

    """ fill in the high_res_neural time course"""

    for hr_onset, hr_duration, amplitude in zip(high_res_onset_indices,
                                                high_res_duration, ampl):
		hr_onset = int(round(hr_onset))
		hr_duration = int(round(hr_duration))
		high_res_neural[hr_onset:hr_onset + hr_duration] = amplitude

    return (high_res_neural, high_res_times)


def constructing_convo(high_res_neural, high_res_times):
    """
    Parameters
    ----------
    high_res_neural: array
        1D array of high resolution prediction time-course.
    high_res_times: array
        1D array of high resolution time.

    Returns
    -------
    high_res_hemo: array
        1D array of high resolution hemodynamic response.
    """
    tr_divs = 100
    hrf_times = np.arange(0, 30, 1/tr_divs)
    hrf_at_trs = hrf(hrf_times)
    high_res_hemo = np.convolve(high_res_neural, hrf_at_trs)[:len(high_res_neural)]
    high_res_times = high_res_times[np.arange(0, len(high_res_times), 100)]
    high_res_hemo = high_res_hemo[np.arange(0, len(high_res_hemo), 100)]
    return (high_res_hemo, high_res_times)
