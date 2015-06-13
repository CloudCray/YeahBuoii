import scipy.fftpack
import os
import numpy as np
import wave
from scipy.signal import argrelextrema


def clip_to_freq(frames):
    sp = np.array([abs(y) for y in np.fft.fft(frames)])
    freq = np.array([abs(x) for x in np.fft.fftfreq(frames.shape[-1])])
    return freq, sp


def local_maxima(ser_x, ser_y):
    m = argrelextrema(ser_y, np.greater)
    x_out = np.array([ser_x[i] for i in m[0]])
    y_out = np.array([ser_y[i] for i in m[0]])
    return x_out, y_out


def smooth_freq(freq, sp_abs, cutoff=500):
    #  y2 = np.fft.irfft(sp_abs)
    w = scipy.fftpack.rfft(sp_abs)
    f = scipy.fftpack.rfftfreq(len(sp_abs), freq[1]-freq[0])
    spectrum = w**2
    cutoff_idx = spectrum < (spectrum.max()/cutoff)
    w2 = w.copy()
    w2[cutoff_idx] = 0
    y2 = scipy.fftpack.irfft(w2)
    return y2
