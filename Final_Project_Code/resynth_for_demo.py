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
        file_name = '_'.join(file.split('_')[1:])  # Common file name
        
        # Get features for synthesis
        f0 = np.load(file)
        mfsc = np.load(mfsc_dir + '/mfsc_' + file_name)
        ap = np.load(ap_dir + '/ap_' + file_name)
        
        # Convert MFSC to SP
        sp = mfsc_to_sp(mfsc)
        # Synthesize the audio
        _synth(file_name, f0, ap, sp)
    
    print('Finished synthesis')
        
def _synth(file, f0, ap, sp, fs=32000, fft_size=2048):      
    file_name = file.split('.')[0] + '.wav'
    y =  pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)
    wavfile.write(file_name, fs, y)

if __name__ == '__main__':
    # Change following 3 lines to specify directories
    fs = 32000
    fft_size = 2048
    f0_dir = './f0'
    ap_dir = './ap'
    mfsc_dir = './mfsc'

    f0 = np.load('C:/Users/Murali/EE599/project/for_demo/f0_original_perfect.npy')
    f0 = f0*2
    ap = np.load('C:/Users/Murali/2018_synth_sing/Wavenet_ap/generated_for_demo_ap.npy')
    ap = ap[20:]
    
    ap_gen_exp = pw.decode_aperiodicity(ap, fs, fft_size)
    # ap = ap[0:1300]
    # f0 = f0[0:1300]
    # mfsc_og = mfsc_og[0:620]
    # np.save("mfsc016_.npy", mfsc_og)

  
    mfsc = np.load('C:/Users/Murali/2018_synth_sing/tensorflow-wavenet/generated_for_demo_mfsc.npy')
    mfsc = mfsc[20:]
    #mfsc = mfsc[20:]
#    pos_idx = np.where(mfsc > 0)
#    mfsc[pos_idx] = 0
    # mfsc = (-1) * np.abs(mfsc)
    # sp_og = mfsc_to_sp(mfsc_og)
    
    sp = mfsc_to_sp(mfsc)

    y =  pw.synthesize(f0, sp, ap_gen_exp, fs, pw.default_frame_period)
    y /= np.max(np.abs(y))
    wavfile.write('resynth_for_demo_1.wav', fs, y)
    #synth(f0_dir, ap_dir, mfsc_dir)
    


