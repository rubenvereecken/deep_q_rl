#!/bin/bash

# Prepare code and logging dir along with common parameters
# source common.sh
cp -r /shared/deep_q_rl /root
cd /root/deep_q_rl
git pull
cd deep_q_rl

SAVE_PATH=/data/logs
mkdir -p $SAVE_PATH

COMMON_PARAMS="--save-path=$SAVE_PATH"

# All defaults will be overwritten by arguments passed to this script
./run_nips.py $COMMON_PARAMS $@
