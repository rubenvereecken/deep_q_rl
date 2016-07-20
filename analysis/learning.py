#!/usr/bin/env ipython

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import csv

from common import *

def plot_loss(learning, outdir, title, ax=None):
    # Allow an axis to be passed in
    if ax is None:
        ax = plt

    steps = np.arange(learning['mean_loss'].size)
    mean_loss_smoothed = smoothing(learning['mean_loss'], 10)
        
    ax.plot(steps, learning['mean_loss'], 'b-', alpha=0.3, markersize=1)
    ax.plot(steps, mean_loss_smoothed, 'b-')
    ax.set_ylabel('mean loss')
    ax.set_xlim(steps[0], steps[-1])
    ax.set_title(title)

    if ax is plt:
        plt.savefig(os.path.join(outdir, 'mean_loss_{}.png'.format(title)))
        plt.clf()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str)
    parser.add_argument('--out', type=str, default='.')
    parser.add_argument('--name', type=str, default='')
    params = parser.parse_args(sys.argv[1:])
    filename = params.filename
    outdir = params.out

    learning = csv_to_dict_of_arrays(filename)
    plot_loss(learning, outdir, name)
