from scipy.signal import gaussian
from scipy.ndimage import filters

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import csv


def csv_to_dict_of_arrays(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        l = list(reader)
        learning = dict.fromkeys(reader.fieldnames)
        # Figure out the types by checking the first row
        types = []
        for d, k in l[0].items():
            try:
                int(k)
                types.append(np.int)
            except ValueError:
                types.append(np.float)
        for i, k in enumerate(learning):
            learning[k] = np.empty(len(l), dtype=types[i])
        for i, row in enumerate(l):
            for k, v in row.items():
                learning[k][i] = v
        return learning


#gaussian filter (running average but closer points have higher weights)
def smoothing(x,window,axis=0):
    filt = gaussian(window,2.)
    return filters.convolve1d(x,filt/np.sum(filt),axis)
