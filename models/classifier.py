import os
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC, OneClassSVM
from scikits.audiolab import Sndfile


def sample_audio(fn, count=10000, start=.5):
    sf = Sndfile(fn)
    frames = sf.read_frames(sf.nframes)
    start_frame = int(sf.nframes * start)
    return frames[start_frame:start_frame+count]

def build_classifier(dir_name, samples=3):
    files = os.listdir(dir_name)
    wav_files = [x for x in files if x.endswith(".wav")]
    dynamic_clf = SVC(kernel='linear')
    classifier_x = []
    classifier_y = []
    for fn in wav_files:
        sf = Sndfile(fn)
        frames = sf.read_frames(sf.nframes)
        class_name = fn.split("_")[1]
        for i in range(samples):
            audio = sample_audio(fn, start=(.5+i*.05))
            classifier_x.append(np.array(audio).flatten())
            classifier_y.append(class_name)
    dynamic_clf.fit(classifier_x, classifier_y)
    return dynamic_clf


def build_nb_classifier(dirname, count=10000):
    files = os.listdir(dirname)
    wav_files = [x for x in files if x.endswith(".wav")]
    clf = GaussianNB()
    classifier_x = []
    classifier_y = []
    for fn in wav_files:
        filefull = os.path.join(dirname, fn)
        sf = Sndfile(filefull)
        frames = sf.read_frames(sf.nframes)
        class_name = fn.split("_")[0]
        audio = frames[:count]
        classifier_x.append(np.array(audio).flatten())
        classifier_y.append(class_name)
    clf.fit(classifier_x, classifier_y)
    return clf