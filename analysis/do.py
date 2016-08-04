#!/usr/bin/env python

import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np

from learning import *
from results import *
from common import *

def roundup(x):
    if x == int(x):
        return x
    else:
        return int(x) + 1

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dirs', type=str, nargs='*')
    parser.add_argument('--out', type=str, default='out')
    # parser.add_argument('--name', type=str, default='')
    parser.add_argument('--no-clean', dest='clean', action='store_false', default=True)
    params = parser.parse_args(sys.argv[1:])
    outdir = params.out

    if params.clean:
        import shutil
        try:
            shutil.rmtree(outdir)
        except:
            pass # dont even care
    try:
        os.makedirs(outdir)
    except Exception as e:
        pass # dont even care


    combo_fig = plt.figure(figsize=(20, 10))
    (ax1, ax2) = combo_fig.add_subplot(1, 2, 1), combo_fig.add_subplot(1, 2, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    dirs = filter(lambda d: os.path.isdir(d), params.dirs)
    print 'About to process {} dirs'.format(len(dirs))
    side = roundup(np.sqrt(len(dirs)))

    overview_size = (40, 40)
    loss_overview = plt.figure(figsize=overview_size)
    q_overview = plt.figure(figsize=overview_size)
    reward_overview = plt.figure(figsize=overview_size)

    for i, d in enumerate(dirs):
        name = os.path.basename(d)
        print 'processing ' + name
        try:
            learning_filename = os.path.join(d, 'learning.csv')
            learning = csv_to_dict_of_arrays(learning_filename)
            results_filename = os.path.join(d, 'results.csv')
            results = csv_to_dict_of_arrays(results_filename)

            params_filename = os.path.join(d, 'parameters.json')

            plot_reward(results, outdir, name, ax)
            fig.savefig(os.path.join(outdir, '{}_avg_reward.png'.format(name)))
            ax.cla()
            plot_q(results, outdir, name, ax)
            fig.savefig(os.path.join(outdir, '{}_avg_q.png'.format(name)))
            ax.cla()

            plot_reward_and_q(results, outdir, name)

            plot_loss(learning, outdir, name, ax1)
            plot_q(results, outdir, name, ax2)
            combo_fig.savefig(os.path.join(outdir, '{}_combined.png'.format(name)))
            ax1.cla()
            ax2.cla()

            overview_loss_axis = loss_overview.add_subplot(side, side, i+1)
            overview_q_axis = q_overview.add_subplot(side, side, i+1)
            overview_reward_axis = reward_overview.add_subplot(side, side, i+1)

            # Plot on overview
            plot_loss(learning, outdir, name, overview_loss_axis)
            plot_q(results, outdir, name, overview_q_axis)
            plot_reward(results, outdir, name, overview_reward_axis)
        except Exception as e:
            print "Skipping {} because of an exception".format(name)

    loss_overview.savefig(os.path.join(outdir, '__overview_loss.png'))
    q_overview.savefig(os.path.join(outdir, '__overview_q.png'))
    reward_overview.savefig(os.path.join(outdir, '__overview_reward.png'))
