import tensorflow as tf
import math
import numpy as np
import pdb

  # loss = np.zeros(NEPOCH) # store the training progress here.
x_data = np.load("mfsc_nitech_jp_song070_f001_003.npy")
previous=[]
for i in range(128):
    print sum(x_data[i])
  