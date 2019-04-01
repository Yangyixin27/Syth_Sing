Weekly updates:
======================

Week 10/28 - 11/3:

Sharada:

- Implemented conversion of acoustic output from WORLD to 60-dimensional log MFSCs
- Discussed interpolation and coarse-coding of F0, and the V/UV model
- Currently implementing F0 interpolation and coarse coding

Other discussions:

- Discussed Wavenet structure and implementing local conditioning
- Held team meetings on 10/29 and 10/30 
- Constrained Gaussian Mixture (CGM) outputs for the mdoel - predict 4 free parameters or 12 GMM parameters?
- Coarse coding F0 and frame position within phoneme
- Aligning phonemes with acoustic features on the frame-level
- Comparing performance of the timbre model by varying the control input (using logF0 (interpolated), using coarse-coded F0 (as in the NPSS paper)

Week 10/15 - 10/19:

* We held a team meeting on Friday, 10/19

James:

- I got the WORLD vocoder working with singing audio. I was able to extract F0, harmonic, and aperiodic acoustic features and resythesize singing from those features. 

- I downloaded 20 isolated vocal tracks from two different artists for training data.

- After struggling with other automated audio-to-score transcription software, I was able to automatically transcribe isolated singing to a MIDI score with reasonably good results using the FlexPitch plugin in Logic. 

Mu:

- I modified and tested the Tensorflow-wavenet implementation on the singing voice dataset MIR-1k. Some [audio samples](https://soundcloud.com/mu-yang-974011976/sets) are generated. 

- I collected 17 isolated vocal tracks from two artists(Coldplay and Adele) and use Gentle to produce phoneme-level alignment json files.

- I plan to add linguistic local conditioning to current wavenet implementation. Also will try to look deeper into the source code to look for approaches replacing raw audio with features generated from WORLD Vocoder.


Yixin: 

- I've trained an "End-to-End Text-To-Speech Synthesis Model" using LJ speech Dataset on HPCC.
- I got in touch with the author of the papper NPSS and he says he'd love to help if we met any problems.

Sharada:

- Acquired the DSD100 dataset containing 100 tracks with isolated vocals. Deleted extraneous audio files (containing mixed and instrumental tracks) and grouped singer-specific tracks, ordered. Uploaded the dataset on HPCC for future use.
- Worked with Yixin on lyric alignment for audio data

- I tried 2 Keras implementation of WaveNet. They are all out of dated but I'll try to make one of them right
- Downloaded 10 mins audio and tried with Gentle. 

Week 10/08 - 10/12:

Sharada:

- I have collected audio data from three disctinct singers (total duration ~40min)
- Investigated note translators(Anthemscore, Sibelius,...) - Anthemscore has a free 30-day trial. While annotating singing vocals, it misses a few notes but is very accurate for instrumentals.
- I plan to start implementing the pitch model of the baseline system.

Mu:

- I have collected audio data and lyrics data, and tried Gentle. demo files are generated. I also explored existing WaveNet implementation. This implementation only implement global conditioning, i.e. speaker identity, and omit linguistic features.

- I plan to address the issue of running Gentle(or other option) in Python. 

- I plan to run the existing WaveNet implementation on an existing singing recording dataset. Generate singing-like audio as starter.

Yixin:
- I have collected audio data and lyrics data from three disctinct singers (total duration ~20min)
- I plan to implement the singer feature extraction code.

James:

- I downloaded the NIT-SONG070-F001 dataset (Japanese nursery rhymes for baseline model testing).

- I'm starting to implement the timbre model of the baseline system with the WORLD vocoder.  

Shiyu:

- Collected around 50 mins audio with corresponding lyrics alignments using gentle from 2 different singers. 
- Reading the Japanese paper and trying to figure out the dataset they are using so we can prepare our data just like that. 



