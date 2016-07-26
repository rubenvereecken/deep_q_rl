#!/bin/bash -l
#PBS -l nodes=1:ppn=4:ivybridge
#PBS -l mem=15gb
#PBS -l walltime=300:00:00
#PBS -M rvereeck@vub.ac.be
#PBS -m e
#PBS -d /u/rvereeck/deep_q_rl/deep_q_rl/
#PBS -o /gpfs/work/rvereeck/out-$PBS_JOBNAME.txt
#PBS -j oe

RESUME_DIR=$1
SAVE_PATH=$RESUME_DIR

if [[ -z $RESUME_DIR ]]; then
  echo "Resume script requires a resume directory"
fi

# Doesnt matter because params get overwritten
SCRIPT=${SCRIPT:="./run_nips.py"}

export THEANO_FLAGS='device=cpu,allow_gc=True,openmp=True,openmp_elemwise_minsize=200000'
export OMP_NUM_THREADS=4

$SCRIPT --save-path $RESUME_DIR --resume

echo "FINISHED" > $SAVE_PATH/state
