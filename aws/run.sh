#!/bin/bash
# Use as ./run_nips.sh --network-type=nips_cpu --log_level=DEBUG

# Prepare code and logging dir along with common parameters
# source common.sh
MOUNT_PATH=/shared
# HEARTBEAT_SCRIPT=/shared/heartbeat.py
# HEARTBEAT_FILE
GIT_BRANCH=${GIT_BRANCH:-"master"}
SCRIPT=${SCRIPT:-"./run_nips.py"}

# cp -r $MOUNT_PATH/deep_q_rl /root
cd /root
git clone https://github.com/rubenvereecken/deep_q_rl
cd deep_q_rl/aws
git checkout $GIT_BRANCH

if [ -z $SAVE_PATH ]; then
  BASE_PATH="$MOUNT_PATH/logs"
  TIME_STR=`python -c "import time;print time.strftime('%d-%m-%Y-%H-%M-%S', time.gmtime())"`
  if [ -z $LABEL ]; then
    SAVE_PATH="$BASE_PATH/$LABEL-$TIME_STR"
  else
    SAVE_PATH="$BASE_PATH/$TIME_STR"
  fi
fi

mkdir -p $SAVE_PATH

COMMON_PARAMS="--save-path=$SAVE_PATH --dont-generate-logdir"
echo "Saving to $SAVE_PATH"

# python HEARTBEAT_SCRIPT -f HEARTBEAT_FILE &

pwd
# All defaults will be overwritten by arguments passed to this script
$SCRIPT $COMMON_PARAMS $@

# Finished the heartbeat
# kill 0

echo "FINISHED" > "$SAVE_PATH/state"
