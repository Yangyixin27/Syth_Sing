# -*- coding: utf-8 -*-
"""
Created on Sat Nov  3 13:15:59 2018

@author: sharada

Converts spectral envelope (from WORLD) to 60-dimensional log MFSCs
These are used as inputs for the Timbre model

NOTE: Requires pysptk
"""
import pysptk
import numpy as np

def specenv2mfsc(sp, order = 60, alpha = 0.45):
    sp = np.abs(sp) # Making sure it is a power spectrum
    
    mc = pysptk.conversion.sp2mc(sp, order-1, alpha)    # Returns 60-dimensional MFSC
    
    logmc = np.log(mc)
    
    return logmc