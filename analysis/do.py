#!/usr/bin/env python
import os
import sys
import argparse
import matplotlib.pyplot as plt
import numpy as np
import re
import json

from learning import *
from results import *
from common import *

def roundup(x):
    if x == int(x):
        return x
    else:
        return int(x) + 1

BLACKLISTED_PARAMS = ['save_path', 'deterministic', 'steps_per_test', 'experiment_prefix',
                        'log_level', 'progress_frequency', 'generate_logdir', 'resume', 'epochs']

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('dirs', type=str, nargs='*')
    parser.add_argument('--out', type=str, default='out')
    # parser.add_argument('--name', type=str, default='')
    parser.add_argument('--no-clean', dest='clean', action='store_false', default=True)
    parser.add_argument('--no-individual', dest='individual', action='store_false', default=True)
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

    r = re.compile('(?P<experiment>.*?)' +
                   '([_-]+rep.(?P<rep>[0-9]+))?' +
                   '([_-]+(?P<date>[0-9]{2}-[0-9]{2}.*?))?$')

    combo_fig = plt.figure(figsize=(20, 10))
    (ax1, ax2) = combo_fig.add_subplot(1, 2, 1), combo_fig.add_subplot(1, 2, 2)
    fig = plt.figure()
    ax = fig.add_subplot(111)

    dirs = filter(lambda d: os.path.isdir(d), params.dirs)
    print 'About to process {} dirs'.format(len(dirs))

    repeated = {}
    learning_dict = {}
    results_dict = {}

    params_union = {}
    params_dict = {}

    # First gather all data and store it in dicts
    for i, d in enumerate(dirs):
        name = os.path.basename(d)
        try:
            learning_filename = os.path.join(d, 'learning.csv')
            learning = csv_to_dict_of_arrays(learning_filename)
            results_filename = os.path.join(d, 'results.csv')
            results = csv_to_dict_of_arrays(results_filename)

            match = r.match(name).groupdict()
            name = match['experiment']

            params_filename = os.path.join(d, 'parameters.json')
            try:
                with open(params_filename, 'r') as f:
                    exp_params = json.load(f)
                    params_dict[name] = exp_params
                    for k, v in exp_params.viewitems():
                        params_union[k] = params_union.get(k, set())
                        params_union[k].add(v)
            except Exception as e:
                print e.message
                print 'Skipping', name

            if match['rep']:
                print name, ' #' + match['rep']
                repeated[name] = repeated.get(name, [])
                repeated[name].append(results)
            else:
                print name
                learning_dict[name] = learning
                results_dict[name] = results
        except IOError as e:
            print "Skipping {}".format(name)
        except Exception as e:
            print type(e)
            print e.message
            print "Skipping {} because something went wrong".format(name)


    # Some experiments were repeated - average those
    for k, v in repeated.iteritems():
        num_reps = len(v)
        name = k
        min_num_epochs = min([len(v[i]['epoch']) for i in range(num_reps)])
        num_epochs = min_num_epochs
        rewards = np.empty((num_reps, num_epochs))
        qs = np.empty((num_reps, num_epochs))

        for i, d in enumerate(v):
            rewards[i] = d['reward_per_episode'][:num_epochs]
            qs[i] = d['mean_q'][:num_epochs]

        mean_rewards = np.mean(rewards, axis=0)
        mean_qs = np.mean(qs, axis=0)

        # Add it to results just like any other
        results_dict[name] = {
                'epoch': np.arange(1, num_epochs+1),
                'reward_per_episode': mean_rewards,
                'mean_q': mean_qs,
        }

    num_results = len(results_dict.keys())
    side = roundup(np.sqrt(num_results))
    overview_size = (side*7.5, side*7.5)
    # loss_overview = plt.figure(figsize=overview_size)
    q_overview = plt.figure(figsize=overview_size)
    reward_overview = plt.figure(figsize=overview_size)

    for i, (name, results) in enumerate(results_dict.iteritems()):
        deviant_params = {k:v for k,v in params_dict[name].viewitems() \
                          if len(params_union[k]) > 1 and k not in BLACKLISTED_PARAMS}

        reps = len(repeated.get(name, []))
        title = ('mean ({}) '.format(reps) if reps > 0 else '') + name
        if len(deviant_params) > 0:
            title = title + '\n' + '\n'.join(['{} = {}'.format(k,v) for k,v in deviant_params.iteritems()])

        if params.individual:
            plot_reward(results, outdir, title, ax)
            fig.savefig(os.path.join(outdir, '{}_avg_reward.png'.format(name)))
            ax.cla()
            plot_q(results, outdir, title, ax)
            fig.savefig(os.path.join(outdir, '{}_avg_q.png'.format(name)))
            ax.cla()

            # TODO temporarily broken bc don't care
            # plot_reward_and_q(results, outdir, title)

            # plot_loss(learning, outdir, title, ax1)
            plot_q(results, outdir, title, ax2)
            combo_fig.savefig(os.path.join(outdir, '{}_combined.png'.format(name)))
            ax1.cla()
            ax2.cla()

        # overview_loss_axis = loss_overview.add_subplot(side, side, i+1)
        overview_q_axis = q_overview.add_subplot(side, side, i+1)
        overview_reward_axis = reward_overview.add_subplot(side, side, i+1)

        # Plot on overview
        # plot_loss(learning, outdir, title, overview_loss_axis)
        plot_q(results, outdir, title, overview_q_axis)
        plot_reward(results, outdir, title, overview_reward_axis)


    # loss_overview.savefig(os.path.join(outdir, '__overview_loss.png'))
    q_overview.tight_layout()
    q_overview.savefig(os.path.join(outdir, '__overview_q.png'))
    reward_overview.tight_layout()
    reward_overview.savefig(os.path.join(outdir, '__overview_reward.png'))
