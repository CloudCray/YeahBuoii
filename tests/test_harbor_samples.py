from models.classifier import build_classifier, sample_audio

import os
import time

train_dir = r"../audio/train_samples"
test_dir = r"../audio/test_samples"

FRAMES = 44100
samples = []

classifier = build_classifier(train_dir, 10, )

tests = 0

for f in os.listdir(test_dir):
    if f.endswith(".wav"):
        fn = os.path.join(test_dir, f)

        sample = sample_audio(fn, 10000, 0)

        p = classifier.predict_sample(sample)

        print("##################################")
        tests += 1
        print("Untrained sample {0}".format(str(tests)))
        print("Processing '{0}'".format(f))
        time.sleep(1)
        print("  - Result: {0}".format(p[0]))