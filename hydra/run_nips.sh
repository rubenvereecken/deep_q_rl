#!/bin/bash -l
#PBS -l nodes=1:ppn=1
#PBS -M rvereeck@vub.ac.be
#PBS -m abe

TIME_STR=`python -c "import time; print time.strftime('%d-%m-%Y_%H-%M-%S')"` 

# Label out directory
LABEL=${LABEL}
if [[ ! -z $LABEL ]]; then
  SAVE_PATH="${TMPDIR}/${LABEL}-${TIME_STR}"
else 
  SAVE_PATH="${TMPDIR}/out-${TIME_STR}"
fi

mkdir -p $SAVE_PATH
echo "Saving to $SAVE_PATH"

PARAMS="--save-path=$SAVE_PATH -r ${ROM} --network-type=${NETWORK_TYPE} --log_level=DEBUG"

echo "Running job on $HOST - " `date`

# All defaults will be overwritten by arguments passed to this script
./deep_q_rl/run_nips.py $PARAMS $@

cp -r $SAVE_PATH $WORKDIR
