__author__ = 'Cloud'
import subprocess
import os

path = "../audio"
os.chdir(path)
files = os.listdir(path)

def output_cl(fn):
    s = "sox {0} {1}".format(fn, fn.lower().replace(".wav", ".flac"))
    return s

for f in files:
    s = output_cl(f)
    subprocess.call(s)