from models.classifier import build_classifier, sample_audio

import os

dirname = r"../audio/samples"

FRAMES = 44100
samples = []

classifier = build_classifier(dirname, 10, "svc")

for f in os.listdir(dirname):
    if f.endswith(".wav"):
        fn = os.path.join(dirname, f)

        sample = sample_audio(fn, 10000, 0)

        p = classifier.predict_sample(sample)

        print("{0}: {1}".format(f, p[0]))