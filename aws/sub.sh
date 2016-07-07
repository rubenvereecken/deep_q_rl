#!/bin/bash

export JOB_NAME=${JOB_NAME:-`python -c "import time;print str(int(time.time()))[-6:]"`}
export JOB_NAME="job-$JOB_NAME"
echo "Job ID ${JOB_NAME}"
qsub -N $JOB_NAME -V run.sh $@
