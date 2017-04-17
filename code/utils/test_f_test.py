""" Tests for f_test module
Run with:
    nosetests test_f_test.py
"""

import numpy as np
import f_test as f
from nose.tools import assert_equal
from numpy.testing import assert_almost_equal, assert_array_equal

X = [[1, 4, 1, 2, 1],
	[2, 7, 5, 5, 9],
	[3, 5, 7, 5, 2],
	[4, 8, 0, 5, 3],
	[5, 4, 3, 6, 5],
	[6, 6, 6, 5, 7],
	[2, 8, 6, 7, 8],
	[7, 2, 7, 6, 2],
	[8, 1, 5, 7, 4],
	[9, 5, 8, 6, 7],
	[7, 7, 8, 5, 4],
	[0, 5, 0, 4, 9],
	[0, 1, 2, 4, 2],
	[8, 8, 6, 9, 0],
	[3, 4, 5, 5, 4]]
y = [3,6,4,8,4,5,9,4,1,4,6,7,2,8,4]
betas = [-0.15943, 0.78309, -0.13022, 0.41716, 0.03324]
np_f_value = np.asarray([1.59113, 50.01170, 1.12854, 6.71188, 0.14581])
np_p_value = np.asarray([0.23580, 0.00003,  0.31307, 0.02692, 0.71056])
f_vl, p_vl = f.f_test(y, X, betas)
f_f_value = []
for fst in f_vl:
	f_f_value.append(round(fst,5))
f_f_value = np.asarray(f_f_value)
f_p_value = []
for pvl in p_vl:
	f_p_value.append(round(pvl,5))
f_p_value = np.asarray(f_p_value)

def test_f():
	assert_array_equal(np_f_value, f_f_value)
def test_p():
	assert_array_equal(np_p_value, f_p_value)

