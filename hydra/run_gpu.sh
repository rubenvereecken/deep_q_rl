#!/bin/bash -l
#PBS -l nodes=1:ppn=4:gpus=1:gpgpu
#PBS -l mem=15gb
#PBS -l walltime=300:00:00
#PBS -M rvereeck@vub.ac.be
#PBS -m e
#PBS -d /u/rvereeck/deep_q_rl/deep_q_rl/
#PBS -o /gpfs/work/rvereeck/out-$PBS_JOBNAME.txt
#PBS -j oe

# Load Cuda module
module load CUDA/7.5.18

TIME_STR=`python -c "import time; print time.strftime('%d-%m-%Y_%H-%M-%S')"` 
SCRIPT=${SCRIPT:="./run_nips.py"}
NETWORK_TYPE=${NETWORK_TYPE:-nips_cudnn}

# Label out directory
LABEL=${LABEL:=$PBS_JOBNAME}
if [ ! -z $LABEL ]; then
  SAVE_PATH="${WORKDIR}/${LABEL}-${ROM}-${TIME_STR}"
else 
  SAVE_PATH="${WORKDIR}/${ROM}-${TIME_STR}"
fi

mkdir -p $SAVE_PATH
echo "Saving to $SAVE_PATH"

ROM=${ROM:="space_invaders"}

PARAMS="--save-path=$SAVE_PATH -r ${ROM} --log_level=DEBUG --dont-generate-logdir --deterministic=False"
if [ ! -z $NETWORK_TYPE ]; then
  PARAMS="$PARAMS --network-type=$NETWORK_TYPE"
fi

echo "Running ${ROM} with ${NETWORK_TYPE} on $HOST - " `date`
echo "RUNNING" > $SAVE_PATH/state

trap "echo INTERRUPTED > $SAVE_PATH/state" INT TERM SIGINT SIGTERM

# All defaults will be overwritten by arguments passed to this script
export THEANO_FLAGS='device=gpu,allow_gc=True,floatX=float32,openmp=True'
#export THEANO_FLAGS='device=gpu,allow_gc=True,floatX=float32'
export OMP_NUM_THREADS=4

# Call the experiment
$SCRIPT $PARAMS $@

echo "FINISHED" > $SAVE_PATH/state
