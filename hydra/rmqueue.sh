#!/bin/bash
STATE=${1:-"R"}
pids="$(./hydra/pids.sh $STATE queue)"
num="$(echo $pids | tr ' ' '\n' | wc -l)"
qdel $pids
echo "Stopped $num $STATE queue jobs"
