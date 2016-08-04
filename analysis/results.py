#!/usr/bin/env python

import sys
import os
import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse

from common import *

def plot_reward(results, outdir, title, ax):
    ax.plot(results['epoch'], results['reward_per_episode'], 'b-')
    ax.plot(results['epoch'], smoothing(results['reward_per_episode'], 10), 'r-')
    ax.set_ylabel('Average reward per epoch', color='b')
    ax.set_title(title)

    # plt.savefig(os.path.join(outdir, 'avg_reward_{}.png'.format(title)))
    # plt.clf()


def plot_q(results, outdir, title, ax):
    ax.plot(results['epoch'], results['mean_q'], 'b-')
    ax.plot(results['epoch'], smoothing(results['mean_q'], 10), 'r-')
    ax.set_ylabel('Average Q per epoch', color='b')
    ax.set_title(title)

    # plt.savefig(os.path.join(outdir, 'avg_q_{}.png'.format(title)))
    # plt.clf()

def plot_reward_and_q(results, outdir, title):
    fig, ax1 = plt.subplots()
    ax1.plot(results['epoch'], results['reward_per_episode'], 'b-')
    ax1.set_xlabel('epoch')
    # Make the y-axis label and tick labels match the line color.
    ax1.set_ylabel('Average reward per episode', color='b')
    for tl in ax1.get_yticklabels():
        tl.set_color('b')

    ax2 = ax1.twinx()
    ax2.plot(results['epoch'], results['mean_q'], 'r-')
    ax2.set_ylabel('average Q', color='r')
    for tl in ax2.get_yticklabels():
        tl.set_color('r')
    plt.title(title)
    plt.savefig(os.path.join(outdir, 'avg_reward_vs_q_{}.png'.format(title)))
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('filename', type=str)
    parser.add_argument('--out', type=str, default='.')
    parser.add_argument('--name', type=str, default='')
    params = parser.parse_args(sys.argv[1:])
    filename = params.filename
    outdir = params.out

    results = csv_to_dict_of_arrays(filename)
    plot_reward(results, outdir, params.name)
    plot_q(results, outdir, params.name)
    plot_reward_and_q(results, outdir, params.name)





