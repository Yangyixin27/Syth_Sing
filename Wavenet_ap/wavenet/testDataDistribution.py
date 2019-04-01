import numpy as np 
import matplotlib.pyplot as plt 
from testCMG_v2 import norm_data

mfsc = np.load("./mfsc_nitech_jp_song070_f001_003.npy")

mfsc = norm_data(mfsc)

data = mfsc[:64, 0]
# print(data.shape)

plt.plot(data)
plt.show()