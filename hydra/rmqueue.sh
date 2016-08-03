#!/bin/bash
STATE=${1:-"R"}
pids="$(./hydra/pids.sh $STATE queue)"
num="$(echo $pids | wc -w)"
echo $pids
qdel $pids
echo "Stopped $num $STATE queue jobs"
