#!/bin/bash

# Prepare code and logging dir along with common parameters
# source common.sh
MOUNT_PATH=/shared
cp -r $MOUNT_PATH/deep_q_rl /root
cd /root/deep_q_rl
git pull
cd deep_q_rl

SAVE_PATH="$MOUNT_PATH/logs"
mkdir -p $SAVE_PATH

COMMON_PARAMS="--save-path=$SAVE_PATH"
echo "Saving to $SAVE_PATH"

# All defaults will be overwritten by arguments passed to this script
./run_mine.py $COMMON_PARAMS $@
