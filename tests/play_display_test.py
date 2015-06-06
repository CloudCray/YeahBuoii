import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from pylab import *

from sklearn.naive_bayes import GaussianNB

from scikits.audiolab import Sndfile

from pprint import pprint

import os

dirname = r"..\audio\samples"

FRAMES = 44100
samples = []

def top_100_f(sample):
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

dynamic_clf = GaussianNB()
dynamic_clf.fit(classifier_x, classifier_y)


import winsound
ion()
fig = plt.figure()

f = "ship_1.wav"
fn = os.path.join(dirname, f)
sf = Sndfile(fn)
frames = sf.read_frames(sf.nframes)
sample = frames[:FRAMES]

p = dynamic_clf.predict(top_100_f(sample))

ion()
fig = plt.figure()
ax = fig.add_subplot(111)
title(p[0])
plot(sample)
draw()

winsound.PlaySound(fn, winsound.SND_FILENAME)
print("{0}: {1}".format(f, p[0]))


f = "whale_1.wav"
fn = os.path.join(dirname, f)
sf = Sndfile(fn)
frames = sf.read_frames(sf.nframes)
sample = frames[:FRAMES]

p = dynamic_clf.predict(top_100_f(sample))

fig = plt.figure()
ax = fig.add_subplot(111)
title(p[0])
plot(sample)
draw()

winsound.PlaySound(fn, winsound.SND_FILENAME)
print("{0}: {1}".format(f, p[0]))


f = "whale_13.wav"
fn = os.path.join(dirname, f)
sf = Sndfile(fn)
frames = sf.read_frames(sf.nframes)
sample = frames[:FRAMES]

p = dynamic_clf.predict(top_100_f(sample))

fig = plt.figure()
ax = fig.add_subplot(111)
title(p[0])
plot(sample)
draw()

winsound.PlaySound(fn, winsound.SND_FILENAME)
print("{0}: {1}".format(f, p[0]))


f = "ship_14.wav"
fn = os.path.join(dirname, f)
sf = Sndfile(fn)
frames = sf.read_frames(sf.nframes)
sample = frames[:FRAMES]

p = dynamic_clf.predict(top_100_f(sample))

ion()
fig = plt.figure()
ax = fig.add_subplot(111)
title(p[0])
plot(sample)
draw()

winsound.PlaySound(fn, winsound.SND_FILENAME)
print("{0}: {1}".format(f, p[0]))


f = "whale_10.wav"
fn = os.path.join(dirname, f)
sf = Sndfile(fn)
frames = sf.read_frames(sf.nframes)
sample = frames[:FRAMES]

p = dynamic_clf.predict(top_100_f(sample))

fig = plt.figure()
ax = fig.add_subplot(111)
title(p[0])
plot(sample)
draw()

winsound.PlaySound(fn, winsound.SND_FILENAME)
print("{0}: {1}".format(f, p[0]))