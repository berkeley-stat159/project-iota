""" Tests for reshape function in linear_modeling module

Run with:

    nosetests test_reshape.py
"""

import numpy as np

from linear_modeling import reshape

from nose.tools import assert_equal

from numpy.testing import assert_almost_equal, assert_array_equal



def test_reshape():
    # We make a fake 4D image
    shape_3d = (2, 3, 4)
    V = np.prod(shape_3d)
    T = 10  # The number of 3D volumes
    # Make a 2D array that we will reshape to 4D
    arr_2d = np.random.normal(size=(V, T))
    differences = np.diff(arr_2d, axis=1)
    exp_rms = np.sqrt(np.mean(differences ** 2, axis=0))
    # Reshape to 4D and run function
    arr_4d = np.reshape(arr_2d, shape_3d + (T,))
    actual_rms = vol_rms_diff(arr_4d)
    assert_almost_equal(actual_rms, exp_rms)
