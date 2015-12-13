from __future__ import division
import numpy as np
#### Discrete cosine transform
def dct(r, t, N):
    """
    Implementing highpass filtering using Discrete Cosine Transformation (DCT)
    input:

    output:



    """
    f = np.sqrt(2/N)*(np.cos(r* np.pi* t/N))

    return f

##### Gaussian 1D signal smoothing
def gaussian_smooth(x,window_len=11, std = 1):
    """smooth the data using a window with requested size.

    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal
    (with the window size) in both ends so that transient parts are minimized
    in the beginning and end part of the output signal.

    input:
        x: the input signal
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal

    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    """

    if x.ndim != 1:
        raise ValueError, "smooth only accepts 1 dimension arrays."

    if x.size < window_len:
        raise ValueError, "Input vector needs to be bigger than window size."

    if window_len<3:
        return x

    s=np.r_[x[window_len-1:0:-1],x,x[-1:-window_len:-1]]

    w=signal.gaussian(window_len, std)

    y=np.convolve(w/w.sum(),s,mode='valid')

    return y


