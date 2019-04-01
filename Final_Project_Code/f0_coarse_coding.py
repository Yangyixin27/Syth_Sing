# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:43:32 2018

@author: sharada

Interpolating F0 values for unvoiced frames
Converting F0 to cents
Coarse coding based on F0 (cents)

NOTE: Needs the location of REFINED F0 .npy files. Change location in line 114
"""

import numpy as np
import pyworld as pw
import scipy.io.wavfile as wavfile
import os
from shutil import rmtree

# Get reference log-f0; to be used in conversion to cents
def get_reference_f0(folder):
    # Find max_f0 and min_f0, assign the mean as the reference
    max_f0 = -1000
    min_f0 = 1000   # As we want nonzero min and max
    
    files = os.listdir(folder)
    
    for file in files:
        curr_f0 = np.load(file)
        curr_f0 = curr_f0[np.nonzero(curr_f0)] 
        curr_f0 = np.log2(curr_f0)   # Taking reference F0 in the log scale
        
        if np.max(curr_f0) > max_f0:
            max_f0 = np.max(curr_f0)
        if np.min(curr_f0) < min_f0:
            min_f0 = np.min(curr_f0)
    
    ref_f0 = 2**((max_f0+min_f0)/2)  # Take the mean as the reference
    max_f0 = 2**max_f0
    min_f0 = 2**min_f0
    
    return ref_f0, max_f0, min_f0

"""
Interpolates unvoiced frames
"""
def fill_uv(folder):
    """
    As we are working on already interpolated F0 values, only need to fill in stray zeros
    """ 
    files = os.listdir(folder)
    
    for file in files:
        f0 = np.load(folder + '/' + file)
        indices = np.where(f0 == 0)[0] 
        # If the first index is zero
        if(indices[0] == 0):
            i = 0
            while f0[i] == 0:
                i = i + 1
            f0[0] = f0[i]   # Replace it with the nearest nonzero value
            indices = np.where(f0 == 0)[0]  # Check again
        
        while(indices.size != 0):
            f0[indices] = f0[indices-1]
            indices = np.where(f0 == 0)[0]
        
        np.save(file, f0)

# Converts logf0 to cents
def f0_to_cents(f0_hz, ref_f0):
    # Take log and convert to cents using formula
    f0_cents = 1200 * np.log2(f0_hz/ref_f0)
    return f0_cents

# Coarse codes f0
def f0_coarse_code(folder, num_states = 3):
    
    # Create folder for storing coarse coded F0
    if os.path.isdir('f0coded'):
        rmtree('f0coded')
    os.mkdir('f0coded')
    
    ref_f0, max_f0, min_f0 = get_reference_f0(folder)
    # Convert the extremes to cents
    max_f0_cents = f0_to_cents(max_f0, ref_f0)
    min_f0_cents = f0_to_cents(min_f0, ref_f0)
       
    files = os.listdir(folder)
    
    for file in files:
        file_name = '_'.join(file.split('_')[1:])  # Common file name
        curr_f0 = np.load(file)
        f0_cents = f0_to_cents(curr_f0, ref_f0) # Convert f0 of current file to cents
        
        # Empty coarse coded matrix
        coarse_code_f0 = np.zeros((len(f0_cents), num_states))
    
        for i in range(len(f0_cents)):
            # Calculate "weight" of each F0 value based on reference and extremeties
            if f0_cents[i] < 0:
                r = (f0_cents[i] - min_f0_cents)/(-min_f0_cents)
                coarse_code_f0[i, 1] = r
                coarse_code_f0[i, 0] = 1-r
            else:
                r = f0_cents[i]/max_f0_cents
                coarse_code_f0[i, 2] = r
                coarse_code_f0[i, 1] = 1-r
        # Save the coarse coded F0
        np.save('./f0coded/coldplay_' + file_name, coarse_code_f0)
        print('Finished coarse coding ' + file)

if __name__ == '__main__':
   
    # Folder containing refined (interpolated) F0
    file_path = './f0'
    
    # Fill stray zero values
    fill_uv(file_path)
    print('Interpolated all f0 values')
    
    # Coarse code all F0 files in the folder
    f0_coarse_code(file_path)
       