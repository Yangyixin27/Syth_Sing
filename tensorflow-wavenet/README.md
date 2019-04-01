# Wavenet
## Log for my progress in wavenet task.
### Update
**Now the experiment using `.npy` seems to be succesful. F0 and acous_feat are aligned**

**Issues:**

- namefile pattern. Now path to F0.npy is hard-coded
- acous_feat_dim and F0_dim are hard-coded now. why can't they be represented? tf issue?
- figure out why there is a gap of 13

=====

- `Wavenet-original`: vanilla implementation by ibab.
- `Wavenet-modified`: modified for very basic MIR-1K synthesis.
- `Wavenet-gc`: commented version of Eavenet-modified. Saved here just for reference purpose.
- `Wavenet-gc-datasetAPI`: trying to use Dataset API

**I gave up implementing the datasetAPI version. Now thinking about modifying the original thread and queue version. What I did include:**

- I disabled coord in train.py and audio reader.py
- I modified several functions in audio_reader.py
- Try to use `.npy` file for some vanilla experiment and modified corresponding parts in the code. but didn't get to that step.

**The reason is that the data iteration happens in the audio_reader.py file, and I don't know how to replace this using dataset API. Otherwise, there will be a big change in the code structure in `train.py`.**

