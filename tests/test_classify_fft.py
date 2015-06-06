import matplotlib
import numpy as np
import matplotlib.pyplot as plt

from sklearn.svm import SVC

from scikits.audiolab import Sndfile

from pprint import pprint

import os

dirname = r"G:\Dropbox\Projects\Engineering\YeahBuoii\audio\samples"

FRAMES = 44100
samples = []

def top_100(sample):
        sp = np.fft.fft(sample)
        freq = np.fft.fftfreq(sample.shape[-1])

        max_y = max(sp.real)

        top_100 = sorted(sp.real)[-100]

        i = list(sp.real).index(max_y)
        max_x = abs(freq[i])

        i100 = list(sp.real).index(top_100)
        top_100_x = abs(freq[i100])
        return np.array([max_x, top_100_x]).flatten()

for f in os.listdir(dirname):
    if f.endswith(".wav"):
        fn = os.path.join(dirname, f)
        sf = Sndfile(fn)
        frames = sf.read_frames(sf.nframes)
        sample = frames[:FRAMES]

        sp = np.fft.fft(sample)
        freq = np.fft.fftfreq(sample.shape[-1])

        max_y = max(sp.real)

        top_100 = sorted(sp.real)[-100]

        i = list(sp.real).index(max_y)
        max_x = abs(freq[i])

        i100 = list(sp.real).index(top_100)
        top_100_x = abs(freq[i100])
        # freq_sort = sorted(freq)
        # top_100 = freq_sort[-100]
        # max_f = max(vals)
        # min_f = min(vals)

        samples.append((f, sample, max_x, top_100_x))


classifier_x = [np.array([x[2], x[3]]).flatten() for x in samples]
classifier_y = [x[0].split("_")[0] for x in samples]

dynamic_clf = SVC(kernel="linear")
dynamic_clf.fit(classifier_x, classifier_y)