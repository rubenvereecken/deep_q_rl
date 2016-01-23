#!/bin/bash

# Prepare code and logging dir along with common parameters
source common.sh

# All defaults will be overwritten by arguments passed to this script
./run_nips.py $COMMON_PARAMS $@
