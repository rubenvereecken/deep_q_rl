import os
import time
import argparse
import sys

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', default=None)
    parser.add_argument('-i', '--interval', default=1, help='in seconds')
    params = parser.parse_args(sys.argv[1:])

    if params.file is None:
        print "Supply a file"
        sys.exit(-1)

    while True:
        os.utime(params.file, None)
        time.sleep(params.interval)


