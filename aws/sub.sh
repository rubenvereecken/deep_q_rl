#!/bin/bash

export JOB_NAME=${JOB_NAME:-job-`python -c "import time;print str(int(time.time()))[-6:]"`}
qsub -N $JOB_NAME -V run.sh $@
