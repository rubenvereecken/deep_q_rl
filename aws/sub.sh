#!/usr/bin/bash

export JOB_NAME=`python -c "import time;print int(time.time())"`
echo "Job ID ${JOB_NAME}"
qsub -N $JOB_NAME -V run.sh $@
