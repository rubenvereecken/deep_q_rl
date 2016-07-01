#!/bin/bash -l
#PBS -l nodes=1:ppn=1
#PBS -M rvereeck@vub.ac.be
#PBS -m abe

# Label out directory
LABEL=${LABEL:=out}

# Not used atm
TIME_STR=`python -c "import time; print time.strftime('%d-%m-%Y_%H-%M-%S')"` 
# SAVE_PATH="${TMPDIR}/out/TIME_STR"

SAVE_PATH="${WORKDIR}/${LABEL}"
mkdir -p $SAVE_PATH

PARAMS="--save-path=$SAVE_PATH -r ${ROM} --network-type=${NETWORK_TYPE} --log_level=DEBUG"
echo "Saving to $SAVE_PATH"

echo "Running job on $HOST - " `date`

# All defaults will be overwritten by arguments passed to this script
./deep_q_rl/run_nips.py $PARAMS $@

