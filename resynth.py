# -*- coding: utf-8 -*-
"""
Created on Sat Nov 10 11:46:21 2018

@author: sharada

Synthesize from .npy files using WORLD

Assumes MFSCs, F0 and APs are stored as .npy files in ./mfsc, ./f0 and ./ap
To change directories, modify lines 53-55
"""
import numpy as np
import scipy.fftpack as sfft
import pysptk
import pyworld as pw
import scipy.io.wavfile as wavfile
import os

def mfsc_to_sp(mfsc, alpha=0.45, N=2048):
    mc = sfft.dct(mfsc, norm = 'ortho') # Mel cepstrum
    sp = pysptk.conversion.mc2sp(mc, alpha, N)  # Spectral envelope
    return sp


# Synthesizes from the .npy files in the folder
def synth(f0_dir, ap_dir, mfsc_dir):
    
    files = os.listdir(f0_dir)
    
    for file in files:
        # file_name = file.split('.')[0]
        # file_name = '_'.join(file.split('_')[1:])  # Common file name
        
        # Get features for synthesis
        f0 = np.load(f0_dir + file)
        mfsc = np.load(mfsc_dir + file)
        ap = np.load(ap_dir + file)
        
        ap = pw.decode_aperiodicity(ap, 32000, 2048)
        
        # Convert MFSC to SP
        sp = mfsc_to_sp(mfsc)
        # Synthesize the audio
        _synth(file, f0, ap, sp)
    
    print('Finished synthesis')
        
def _synth(file, f0, ap, sp, fs=32000, fft_size=2048):      
    file_name = file.split('.')[0] + '.wav'
    y =  pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)
    wavfile.write('C:/Users/Murali/EE599/project/wav_synth/' + file_name, fs, y)

if __name__ == '__main__':
    # Change following 3 lines to specify directories
    f0_dir = 'C:/Users/Murali/EE599/project/NIT/NIT/f0ref/'
    ap_dir = 'C:/Users/Murali/EE599/project/NIT/NIT/ap/'
    mfsc_dir = 'C:/Users/Murali/EE599/project/mfsc/'
    
    synth(f0_dir, ap_dir, mfsc_dir)