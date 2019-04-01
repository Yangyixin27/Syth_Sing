from __future__ import division, print_function
import os
from shutil import rmtree
import argparse
import numpy as np
import matplotlib.pyplot as plt
import soundfile as sf
import scipy.io.wavfile as wav
import pyworld as pw
import pysptk
import scipy.fftpack as sfft

# create a parser
parser = argparse.ArgumentParser()
# add arguments
parser.add_argument("-f", "--frame_period", type=float, default=5.0)
parser.add_argument("-s", "--speed", type=int, default=1)

DEFAULT = 0
DIO = 0
DIO_REFINE = 0
HARVEST = 1

def raw2wav():
    # delete .wav directory
    if os.path.isdir('wav'):
        rmtree('wav')
    # create directory for .wav files 
    os.mkdir('wav')

    # find directory of .raw files
    if os.path.isdir('raw'):
        raw_dir = './raw'
        # get list of all .raw files
        raw_files = os.listdir(raw_dir)
        for file in raw_files:  
            raw_file = "./raw/" + file
            file = file.split('.')[0]
            # create corresponding .wav file name
            wav_file = "./wav/" + file + ".wav"
            # convert .raw file to .wav files   
            os.system("ffmpeg -f s16le -ar 48000 -ac 1 -i " + raw_file + " ./wav/temp.wav")
            # resample .wav file to 32kHz
            os.system("sox " + "./wav/temp.wav " + "-c1 " + wav_file + " rate " + "32000")
            os.remove("./wav/temp.wav")
    else:
        print("Can't find .raw file directory!")

def extract_acoustic_features():
    
    # create f0 directory
    if os.path.isdir('f0'):
        rmtree('f0')
    os.mkdir('f0')
    
    # create refined f0 directory
    if os.path.isdir('f0ref'):
        rmtree('f0ref')
    os.mkdir('f0ref')

    # create v/uv directory 
    if os.path.isdir('voicing'):
        rmtree('voicing')
    os.mkdir('voicing')
    
    # create sp directory 
    if os.path.isdir('sp'):
        rmtree('sp')
    os.mkdir('sp')
    
    # create mfsc directory 
    if os.path.isdir('mfsc'):
        rmtree('mfsc')
    os.mkdir('mfsc')

    # create ap directory
    if os.path.isdir('ap'):
        rmtree('ap')
    os.mkdir('ap')

    # read in all files in .wav directory
    if os.path.isdir('wav'):
        wav_dir = './wav'
        wav_files = os.listdir(wav_dir)
        for file in wav_files:
            wav_file = "./wav/" + file
            # read .wav file
            x, fs = sf.read(wav_file)
            # use default options 
            if DEFAULT:
                f0, sp, ap = pw.wav2world(x, fs) 

            else:
                # extract f0           
                f0, t = pw.harvest(x, fs)
                # f0 refinement
                f0_ref = pw.stonemask(x, f0, t, fs)

                # find voice/unvoiced decision 
                uv = (f0 != 0)
                uv = uv.astype(int)
                # extract sp
                sp = pw.cheaptrick(x, f0, t, fs)
                # extract mfsc
                mfsc = get_mfsc(sp)
                # extract ap
                ap = pw.d4c(x, f0, t, fs)

                # reduced ap dimensionality 
                ap_reduced_dim = pw.code_aperiodicity(ap,fs) 
                #n_aper = pw.get_num_aperiodicities(fs)
                #print(n_aper)  

                ####### TESTING #########
                print("Extracting features for " + file)
                file = file.split('.')[0]
                np.save("./f0/f0_" + file + ".npy", f0)
                np.save("./f0ref/f0ref_" + file + ".npy", f0_ref)
                np.save("./voicing/voicing_" + file + ".npy", uv)
                np.save("./sp/sp_" + file + ".npy", sp)
                np.save("./mfsc/mfsc_" + file + ".npy", mfsc)
                np.save("./ap/ap_" + file + ".npy", ap_reduced_dim)

    else:
        print("Can't find .wav file directory!")

# Get 60 log-MFSCs from the spectral envelope
def get_mfsc(sp, alpha=0.45, order=60):
    sp = np.abs(sp)
    mc = pysptk.conversion.sp2mc(sp, order, alpha)    # Mel cepstrum
    mfsc = sfft.idct(mc, n=order, norm='ortho')   # Log mel spectrum
    return mfsc

def main(args):
    raw2wav()
    extract_acoustic_features()

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
