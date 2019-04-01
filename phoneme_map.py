# -*- coding: utf-8 -*-
"""
Created on Tue Nov 27 19:33:40 2018

@author: sharada

To get accurate prev, curr and next phoneme
"""
import numpy as np

FRAME_HOP = 0.005


def get_idxs(intervals, i):
    # Start and end frame index
    start_idx = int(intervals[i, 0] / FRAME_HOP)
    end_idx = int(intervals[i, 1] / FRAME_HOP)
    
    return start_idx, end_idx

def fill_from_beginning(to_be_filled, vector):
    # Vector is what should be filled instead of the first phoneme
    found = True
    ref = np.copy(to_be_filled[0])
    
    j = 0
    max_j = to_be_filled.shape[0]
    
    while(found and j < max_j):
        to_be_filled[j] = vector
        
        # If the next row has a different phoneme
        print(ref)
        print(to_be_filled[j+1])
        print()
        if np.array_equal(to_be_filled[j+1], ref):
            j += 1
        else:
            found = False
    
    return to_be_filled

def fill_from_end(to_be_filled, vector):
    # Vector is what should be filled instead of the first phoneme
    found = True
    ref = np.copy(to_be_filled[-1])
    
    j = to_be_filled.shape[0] - 1   # Get index of last frame
    
    while(found and j >= 0):
        to_be_filled[j] = vector
        
        # If the previous row has a different phoneme
        if not np.array_equal(to_be_filled[j-1], ref):
            found = False
        else:
            j -= 1
    
    return to_be_filled

if __name__ == '__main__':
    
    # Give clip IDs here
    clip_ids = ["004", "007", "037", "010", "004", "004", "004", "004"]
    
    # Give durations here
    intervals = np.array([[1.1, 2.2], [6.5, 7.8], [5.4, 6.8], [15.6, 16.8], [12.0, 13.2], [17.9, 19.0], [21.9, 22.7], [26.4, 27.2]])
    
    # Fetch directories + current/prev/pos/next
    fetch_dir = "C:/Users/Murali/EE599/project/phones/"
    song_prefix = "nitech_jp_song070_f001_"
    
    # Store directories + current/prev/pos/next
    store_dir = "C:/Users/Murali/EE599/project/for_demo/new_phones/"
    
    # Get the file : 3 files to save for each - prev_of_curr, curr, next_of_curr
    # Will also use : prev
      
    prev_phonemes = np.array([]) # Placeholder
    CURRENT = np.array([])
    PREV = np.array([])
    NEXT = np.array([])
    
    for i in range(len(clip_ids)):
        
        # Start and end frame index
        start_idx, end_idx = get_idxs(intervals, i)
        
        # Load current, previous and next phonemes as npy arrays
        curr = np.load(fetch_dir + "current/" + song_prefix + clip_ids[i] + ".npy")
        curr = curr[start_idx:end_idx]
        
        prev_of_curr = np.load(fetch_dir + "prev/" + song_prefix + clip_ids[i] + ".npy")
        prev_of_curr = prev_of_curr[start_idx:end_idx]
        
        next_of_curr = np.load(fetch_dir + "next/" + song_prefix + clip_ids[i] + ".npy")
        next_of_curr = next_of_curr[start_idx:end_idx]
                
        if i == 0:  
            # CURRENT: Store current phonemes
            CURRENT = np.copy(curr)
            
            # PREV: Fill the first unique entry of the prev_phoneme array with all zeros
            unk = np.zeros(prev_of_curr.shape[1])            
            # Starting of store of previous array with all previous phonemes
            print(prev_of_curr[0])
            PREV = np.array(fill_from_beginning(prev_of_curr, unk))
            
            # NEXT: Fill last value of next from first value of next_phonemes           
            # Load current phonemes of next clip
            next_phonemes = np.load(fetch_dir + "current/" + song_prefix + clip_ids[i+1] + ".npy")
            start, _ = get_idxs(intervals, i+1)
            last = next_phonemes[start]
            NEXT = np.array(fill_from_end(next_of_curr, last))
            
        else:
            # CURRENT: Add current phonemes
            CURRENT = np.append(CURRENT, curr, axis = 0)
            
            # PREV: Fill first of previous with last of curr of prev
            first = prev_phonemes[-1]
            PREV = np.append(PREV, fill_from_beginning(prev_of_curr, first), axis = 0)
            
            # NEXT:
            if i == (len(clip_ids) - 1):
                last = np.zeros(next_of_curr.shape[1])
                NEXT = np.append(NEXT, fill_from_end(next_of_curr, last), axis = 0)
            else:
                 # Load current phonemes of next clip
                next_phonemes = np.load(fetch_dir + "current/" + song_prefix + clip_ids[i+1] + ".npy")
                start, _ = get_idxs(intervals, i+1)
                last = next_phonemes[start]
                NEXT = np.append(NEXT, fill_from_end(next_of_curr, last), axis = 0)
            
        
        # Update prev_phonemes array
        prev_phonemes = curr
    
    np.save(store_dir + "current/" + "concatenated.npy" , CURRENT)
    np.save(store_dir + "prev/" + "concatenated.npy" , PREV)
    np.save(store_dir + "next/" + "concatenated.npy" , NEXT)
        
            
            