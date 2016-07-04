import os
import time
import argparse
import sys
import json
import glob
import re

def qstat():
    raw = subprocess.check_output(['qstat'])
    raw = re.sub(r'( )+', ' ', raw)
    raw = re.sub(r'^ +', '', raw)
    lines = raw.split('\n')[2:]
    names = map(lambda l: l[2], lines)
    states = map(lambda l: l[4], lines)
    print names, states
    return names, states

def submit(state_filename, description_filename):
    with open(state_filename) as f:
        state.write('QUEUED')

    with open(description_filename) as f:
        desc = json.load(f)

        parameters = []
        parameters += ['-r', desc.rom]
        parameters += ['--network-type', desc.network]
        parameters += ['--log_level', desc.log]

        if params.args:
            parameters += [desc.args]

        sub.process.Popen(['qsub', '-V', '-jy', '-N', + desc.id_, desc.script] + parameters)


if __name__ == '__main__':
    MOUNT_PATH = '/shared'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--jobdir', default=MOUNT_PATH + '/jobs')
    parser.add_argument('-s', '--script', default='./run_nips')
    parser.add_argument('-i', '--interval', default=5, help='in seconds')
    parser.add_argument('-t', '--timeout', default=5, help='heartbeat timeout')
    params = parser.parse_args(sys.argv[1:])

    while True:
        known_ids, qstates = qstat()

        for description_filename in glob.glob(params.jobdir + '*.desc'):
            desc_name, _ = os.path.splitext(description_filename)
            state_filename = os.path.join(desc_name, '.state')

            if not os.path.exists(state_filename):
                # Submit to the queue and create a state file
                submit(state_filename, description_filename)

            else:
                with open(state_filename, 'r') as f:
                    state = f.read()
                    found = True
                    try:
                        i = known_ids.index(desc.id_)
                        qstate = qstates[i]
                    except ValueError as e:
                        found = False

                    if state == 'QUEUED':
                        if not found:
                            print desc.id_ ' not found why is this imma resubmit'
                            submit(state_filename, description_filename)
                        elif qstate is 'r':
                            with open(state_filename) as f:
                                state.write('RUNNING')
                        elif qstate is 'qw':
                            print desc.id_ + ' still queued'
                    elif state == 'FINISHED':
                        pass # Nothing to see here
                    elif state == 'CANCELLED':
                        print desc.id_ + ' found cancelled'
                    elif state == 'RUNNING':
                        if not found:
                            raise Exception('Alright maybe interrupted anyways')
                        print 'still running strong'
                    elif state == 'INTERRUPTED'
                        if found:
                            raise Exception('Not supposed to be able to find interrupted')
                        # Just resubmit, no worries. That's what I'm here for
                        submit(state_filename, description_filename)

            time.sleep(params.interval)








