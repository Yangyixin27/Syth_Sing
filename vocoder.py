from __future__ import division, print_function
import subprocess
import matplotlib.pyplot as plt
import numpy as np
import os
import soundfile as sf

import pyworld as pw



import os
from shutil import rmtree
import argparse

import numpy as np

import matplotlib      # Remove this line if you don't need them
matplotlib.use('Agg')  # Remove this line if you don't need them
import matplotlib.pyplot as plt

import soundfile as sf
# import librosa
import pyworld as pw

DOWNLOAD = 0
EXTRACT = 0


# youtube links
# ed sheeran songs
youtube_links = {"Thinking_Out_Loud" : "https://www.youtube.com/watch?v=818gWKNXMig",
				"Dive" : "https://www.youtube.com/watch?v=_5TUMYqU2ak"}
				# "Sing": "https://www.youtube.com/watch?v=f816x4FAo50", \
				# "Photograph" : "https://www.youtube.com/watch?v=-H1X7JZwD98", \
				# "ATeam" : "https://www.youtube.com/watch?v=S1B6R0OHy7c", \
				# "Galway_Girl" : "https://www.youtube.com/watch?v=yYlr0QdVZ0g", \
				# "Shape_Of_You" : "https://www.youtube.com/watch?v=yYlr0QdVZ0g", \
				# "Perfect" : "https://www.youtube.com/watch?v=XMqEFuGA2cE",\
				# "Happier" : "https://www.youtube.com/watch?v=TMX-HebGwRE"}


# raw file names
raw_files = {"Thinking_Out_Loud": "Thinking_Out_Loud", \
			"Dive": "Dive"}

# edited file names 
edited_files = {"Thinking_Out_Loud": "Thinking_Out_Loud_edit", \
				"Dive": "Dive_edit"}

# # audio library 
audioLib = ["Thinking_Out_Loud", "Dive"]


if DOWNLOAD: 
	print("Downloading & processing audio files from Youtube...")
	for i in raw_files:
		# download clean speech and noise
		subprocess.run(["youtube-dl", "-x", "--audio-format", "wav", "-o", "{}.%(wav)s".format(raw_files[i]), youtube_links[i]])
		# trim files to 30 sec long
		# resample files to 16kHz
		subprocess.run(["sox", "{}.wav".format(raw_files[i]), "-c1", "{}.wav".format(edited_files[i]), "trim", "00:00", "01:00", "rate", "41000"])

if EXTRACT:
	# train speech
	#fs, x = wavfile.read("./{}.wav".format(edited_files["Thinking_Out_Loud"]))
	x, fs = sf.read("./{}.wav".format(edited_files["Thinking_Out_Loud"]))

	_f0, t = pw.dio(x, fs)    # raw pitch extractor
	# f0 = pw.stonemask(x, _f0, t, fs)  # pitch refinement
	# sp = pw.cheaptrick(x, f0, t, fs)  # extract smoothed spectrogram
	# ap = pw.d4c(x, f0, t, fs)         # extract aperiodicity
	# y = pw.synthesize(f0, sp, ap, fs)

	#Convert speech into features (using default options)
	f0, sp, ap = pw.wav2world(x, fs)


	parser = argparse.ArgumentParser()
	parser.add_argument("-f", "--frame_period", type=float, default=5.0)
	parser.add_argument("-s", "--speed", type=int, default=1)


	EPSILON = 1e-8

	def savefig(filename, figlist, log=True):
	    #h = 10
	    n = len(figlist)
	    # peek into instances
	    f = figlist[0]
	    if len(f.shape) == 1:
	        plt.figure()
	        for i, f in enumerate(figlist):
	            plt.subplot(n, 1, i+1)
	            if len(f.shape) == 1:
	                plt.plot(f)
	                plt.xlim([0, len(f)])
	    elif len(f.shape) == 2:
	        Nsmp, dim = figlist[0].shape
	        #figsize=(h * float(Nsmp) / dim, len(figlist) * h)
	        #plt.figure(figsize=figsize)
	        plt.figure()
	        for i, f in enumerate(figlist):
	            plt.subplot(n, 1, i+1)
	            if log:
	                x = np.log(f + EPSILON)
	            else:
	                x = f + EPSILON
	            plt.imshow(x.T, origin='lower', interpolation='none', aspect='auto', extent=(0, x.shape[0], 0, x.shape[1]))
	    else:
	        raise ValueError('Input dimension must < 3.')
	    plt.savefig(filename)
	    # plt.close()


	def main(args):
	    if os.path.isdir('test'):
	        rmtree('test')
	    os.mkdir('test')

	    x, fs = sf.read("./{}.wav".format(edited_files["Thinking_Out_Loud"]))
	    # x, fs = librosa.load('utterance/vaiueo2d.wav', dtype=np.float64)

	    # 1. A convient way
	    f0, sp, ap = pw.wav2world(x, fs)    # use default options
	    y = pw.synthesize(f0, sp, ap, fs, pw.default_frame_period)

	    # 2. Step by step
	    # 2-1 Without F0 refinement
	    _f0, t = pw.dio(x, fs, f0_floor=50.0, f0_ceil=600.0,
	                    channels_in_octave=2,
	                    frame_period=args.frame_period,
	                    speed=args.speed)
	    _sp = pw.cheaptrick(x, _f0, t, fs)
	    _ap = pw.d4c(x, _f0, t, fs)
	    _y = pw.synthesize(_f0, _sp, _ap, fs, args.frame_period)
	    # librosa.output.write_wav('test/y_without_f0_refinement.wav', _y, fs)
	    sf.write('test/y_without_f0_refinement.wav', _y, fs)

	    # 2-2 DIO with F0 refinement (using Stonemask)
	    f0 = pw.stonemask(x, _f0, t, fs)
	    sp = pw.cheaptrick(x, f0, t, fs)
	    ap = pw.d4c(x, f0, t, fs)
	    y = pw.synthesize(f0, sp, ap, fs, args.frame_period)
	    # librosa.output.write_wav('test/y_with_f0_refinement.wav', y, fs)
	    sf.write('test/y_with_f0_refinement.wav', y, fs)

	    # 2-3 Harvest with F0 refinement (using Stonemask)
	    _f0_h, t_h = pw.harvest(x, fs)
	    f0_h = pw.stonemask(x, _f0_h, t_h, fs)
	    sp_h = pw.cheaptrick(x, f0_h, t_h, fs)
	    ap_h = pw.d4c(x, f0_h, t_h, fs)
	    y_h = pw.synthesize(f0_h, sp_h, ap_h, fs, pw.default_frame_period)
	    # librosa.output.write_wav('test/y_harvest_with_f0_refinement.wav', y_h, fs)
	    sf.write('test/y_harvest_with_f0_refinement.wav', y_h, fs)

	    # Comparison
	    savefig('test/wavform.png', [x, _y, y])
	    savefig('test/sp.png', [_sp, sp])
	    savefig('test/ap.png', [_ap, ap], log=False)
	    savefig('test/f0.png', [_f0, f0])

	    print('Please check "test" directory for output files')


	if __name__ == '__main__':
	    args = parser.parse_args()
	    main(args)












