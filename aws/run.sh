#!/bin/bash
# Use as ./run_nips.sh --network-type=nips_cpu --log_level=DEBUG

# Prepare code and logging dir along with common parameters
# source common.sh
MOUNT_PATH=/shared
# HEARTBEAT_SCRIPT=/shared/heartbeat.py
# HEARTBEAT_FILE
GIT_BRANCH=${GIT_BRANCH:-"master"}
SCRIPT=${SCRIPT:-"run_nips.py"}

# cp -r $MOUNT_PATH/deep_q_rl /root
cd /root
if [ -d "deep_q_rl" ]; then
  cd deep_q_rl
else
  git clone https://github.com/rubenvereecken/deep_q_rl
  cp -r $MOUNT_PATH/roms deep_q_rl
fi
git checkout $GIT_BRANCH
git pull
cd deep_q_rl/deep_q_rl

if [ -z $SAVE_PATH ]; then
  BASE_PATH="$MOUNT_PATH/logs"
  # TIME_STR=`python -c "import time;print time.strftime(\'%d-%m-%Y-%H-%M-%S\', time.gmtime())"`
  TIME_STR=`date "+%d-%m-%Y_%H:%M:%S"`
  if [ -z $LABEL ]; then
    SAVE_PATH="$BASE_PATH/$TIME_STR"
  else
    SAVE_PATH="$BASE_PATH/$LABEL-$TIME_STR"
  fi
fi

mkdir -p $SAVE_PATH

COMMON_PARAMS="--save-path=$SAVE_PATH --dont-generate-logdir"
echo "Saving to $SAVE_PATH"

if [ -z $JOB_NAME ]; then
  echo "Could not find JOB_NAME env var"
  exit -1
fi
# python HEARTBEAT_SCRIPT -f HEARTBEAT_FILE &
echo $JOB_NAME > $SAVE_PATH/job

pwd
# All defaults will be overwritten by arguments passed to this script
./$SCRIPT $COMMON_PARAMS $@

# Finished the heartbeat
# kill 0

echo "FINISHED" > "$SAVE_PATH/state"
