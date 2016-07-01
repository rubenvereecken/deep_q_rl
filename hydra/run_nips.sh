#!/bin/bash -l
#PBS -l nodes=1:ppn=1
#PBS -l walltime=
#PBS -o nips.out

TIME_STR=`python -c "import time; print time.strftime('%d-%m-%Y_%H-%M-%S')"` 
# SAVE_PATH="${TMPDIR}/out/TIME_STR"
SAVE_PATH="${WORKDIR}/out"
mkdir -p $SAVE_PATH

COMMON_PARAMS="--save-path=$SAVE_PATH"
echo "Saving to $SAVE_PATH"

echo "Running job on $HOST - " `date`

# All defaults will be overwritten by arguments passed to this script
/run_nips.py $COMMON_PARAMS $@

