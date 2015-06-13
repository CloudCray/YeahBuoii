import os
import numpy as np
import wave
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from models.processing import clip_to_freq, smooth_freq, local_maxima

class AudioClassifier:
    def __init__(self, sample_size=5, c="gnb"):
        self.c = c
        if c == "gnb":
            self.classifier = GaussianNB()
        elif c == "svc":
            self.classifier = SVC()
        elif c == "lsvc":
            self.classifier = SVC(kernel="linear")
        self.classifier_x = []
        self.classifier_y = []
        self.sample_size = sample_size

    def add_point(self, x, y, refit=True):
        self.classifier_x.append(x)
        self.classifier_y.append(y)
        if refit:
            if self.c == "gnb":
                self.classifier = GaussianNB()
            elif self.c == "svc":
                self.classifier = SVC()
            elif self.c == "lsvc":
                self.classifier = SVC(kernel="linear")
            self.classifier.fit(self.classifier_x, self.classifier_y)

    def predict(self, x):
        return self.classifier.predict(x)

    def predict_sample(self, frames):
        freq, sp = clip_to_freq(frames)
        smooth_sp = smooth_freq(freq, sp, cutoff=600)
        lm_x, lm_y = local_maxima(freq, smooth_sp)
        sorted_maxima = [x for (y, x) in sorted(zip(lm_x, lm_y))]
        if len(sorted_maxima) < 100:
            sorted_maxima += [0 for i in range(100)]
        top_100 = sorted_maxima[:100]
        top_x = np.array(top_100[:self.sample_size]).flatten()
        return self.classifier.predict(top_x)


def sample_audio(fn, count=10000, start=.5):
    wf = wave.open(fn, "rb")
    nframes = wf.getnframes()
    frames = wf.readframes(nframes)
    sample = np.fromstring(frames, "Int16")
    start_frame = int(nframes * start)
    return sample[start_frame:start_frame+count]


def build_classifier(dir_name, sample_size=5, c="gnb"):
    files = os.listdir(dir_name)
    wav_files = [x for x in files if x.endswith(".wav")]
    classifier = AudioClassifier(sample_size, c)
    for fn in wav_files:
        file_full = os.path.join(dir_name, fn)
        frames = sample_audio(file_full, start=0)
        freq, sp = clip_to_freq(frames)
        smooth_sp = smooth_freq(freq, sp, cutoff=600)
        lm_x, lm_y = local_maxima(freq, smooth_sp)
        sorted_maxima = [x for (y, x) in sorted(zip(lm_x, lm_y))]
        if len(sorted_maxima) < 100:
            sorted_maxima += [0 for i in range(100)]
        top_100 = sorted_maxima[:100]
        top_x = np.array(top_100[:classifier.sample_size]).flatten()
        class_name = fn.split("_")[0]
        refit = False
        if len(set(classifier.classifier_y)) > 1:
            refit = True
        classifier.add_point(top_x, class_name, refit)
    return classifier
