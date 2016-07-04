import os
import time
import argparse
import sys
import json
from datetime import datetime

if __name__ == '__main__':
    MOUNT_PATH = '/shared'

    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--jobdir', default=MOUNT_PATH + '/jobs')
    parser.add_argument('-e', '--script', default='./run.sh')
    parser.add_argument('-r', '--rom', required=True)
    parser.add_argument('-n', '--network', default='nip_cudnn')
    parser.add_argument('-l', '--label', default='')
    parser.add_argument('-s', '--savepath', default=MOUNT_PATH + '/logs')
    parser.add_argument('--log', default='DEBUG')
    parser.add_argument('-a', '--args', default=None, help='additional args')
    params = parser.parse_args(sys.argv[1:])

    try:
        os.makedirs(params.jobdir)
    except OSError as ex:
        # Directory most likely already exists
        pass
    
    time_str = time.strftime("%d-%m-%Y-%H-%M-%S", time.gmtime())
    pre = params.label + '-' if params.label else ''
    filename = params.jobdir + '/' + pre + time_str + '.desc'

    if os.path.isfile(filename):
        print "Descriptor already exists"
        sys.exit(-1)

    params.script = os.path.realpath(params.script)
    if not os.path.exists(params.script):
        print "Script to submit to queue does not exist"
        sys.exit(-1)

    with open(filename, 'w') as f:
        json.dump({
            rom: params.rom,
            network: params.network,
            label: pre + time_str,
            log: params.log,
            args: params.args,
            script: params.script,
            id_: datetime.now().microsecond
        }, f)
    
