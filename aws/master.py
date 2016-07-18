import os
import time
import argparse
import sys
import re
import subprocess
import logging

def qstat():
    raw = subprocess.check_output(['qstat'])
    raw = re.sub(r'( )+', ' ', raw)
    trim_leading = lambda x: re.sub(r'^ +', '', x)
    lines = raw.split('\n')[2:]
    lines = map(trim_leading, lines)
    lines = filter(lambda x: len(x) != 0, lines)
    names = map(lambda l: l.split(' ')[2], lines)
    states = map(lambda l: l.split(' ')[4], lines)
    return names, states

if __name__ == '__main__':
    MOUNT_PATH = '/shared'

    parser = argparse.ArgumentParser()
    parser.add_argument('-w', '--watch-dir', dest='watch_dir', default=MOUNT_PATH + '/logs')
    parser.add_argument('-l', '--log-dir', dest='log_dir', default=MOUNT_PATH + '/logs')
    parser.add_argument('-i', '--interval', default=5, help='in seconds')
    # parser.add_argument('-t', '--timeout', default=5, help='heartbeat timeout')
    params = parser.parse_args(sys.argv[1:])
    print 'Watching top-level dir {}'.format(params.watch_dir)

    log = logging.getLogger()
    logFormatter = logging.Formatter('%(asctime)s - %(message)s')
    streamHandler = logging.StreamHandler(sys.stdout)
    streamHandler.setFormatter(logFormatter)
    streamHandler.setLevel(logging.DEBUG)
    fileHandler = logging.FileHandler(params.log_dir + '/master.log')
    fileHandler.addHandler(fileHandler)
    fileHandler.setLevel(logging.INFO)
    log.addHandler(streamHandler)

    while True:
        known_ids, qstates = qstat()
        print known_ids

        for dirname in os.listdir(params.watch_dir):
            dirname = params.watch_dir + '/' + dirname
            if not os.path.isdir(dirname): continue
            with open(dirname + '/job', 'r') as f:
                job = f.read()

            state_filename = os.path.join(dirname, 'state')

            try:
                i = known_ids.index(desc.id_)
                qstate = qstates[i]
                jobname = known_ids[i]
            except ValueError as e:
                qstate = None
                jobname = None

            try:
                with open(state_filename, 'r') as f:
                    state = f.read()
            except IOError as e:
                state = None

            if qstate == 'qw':
                print '{} still queued'.format(jobname)
                pass # QUEUED
            elif qstate == 'r':
                print '{} still running'.format(jobname)
                pass # Cool, it's running
            elif state in ['FINISHED', 'CANCELLED']:
                print 'passing {}; {}'.format(jobname, state)
                pass # Alright I guess that's fine too
            else:
                print '{} appears to have stopped prematurely, restarting'.format(jobname)
                # Should be running, but it isn't
                subprocess.Popen(['bash', 'sub.sh', '-N', jobname,
                    '--save-path', dirname, '--resume'])

        time.sleep(params.interval)
