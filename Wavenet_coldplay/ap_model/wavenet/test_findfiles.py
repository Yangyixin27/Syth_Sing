# from audio_reader import find_files
import fnmatch
import os
import random
import re
# import threading

# import librosa
import numpy as np

def find_files(directory, pattern="*.npy"):   ######################################################
    
    files = []
    for root, dirnames, filenames in os.walk(directory):
        for filename in fnmatch.filter(filenames, pattern):
            files.append(filename)
    return files


files = find_files("/Users/muyang/Desktop/EE599_Speech/2018_synth_sing_proj/Dataset/HTS-demo_NIT-SONG070-F001/data/mfsc/", "*.npy")
print (len(files))