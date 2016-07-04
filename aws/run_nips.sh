#!/bin/bash
# Use as ./run_nips.sh --network-type=nips_cpu --log_level=DEBUG

# Prepare code and logging dir along with common parameters
# source common.sh
MOUNT_PATH=/shared
HEARTBEAT_SCRIPT=/shared/heartbeat.py
# HEARTBEAT_FILE

cp -r $MOUNT_PATH/deep_q_rl /root
cd /root/deep_q_rl
git pull
cd deep_q_rl

SAVE_PATH="$MOUNT_PATH/logs"
mkdir -p $SAVE_PATH

COMMON_PARAMS="--save-path=$SAVE_PATH"
echo "Saving to $SAVE_PATH"

python HEARTBEAT_SCRIPT -f HEARTBEAT_FILE &

# All defaults will be overwritten by arguments passed to this script
./run_nips.py $COMMON_PARAMS $@

# Finished the heartbeat
kill 0

echo "COMPLETED" > HEARTBEAT_FILE
