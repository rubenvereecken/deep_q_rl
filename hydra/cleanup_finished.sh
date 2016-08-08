#!/bin/bash
SSH_STRING="rvereeck@hydra.vub.ac.be"

finished_dirs=`ssh ${SSH_STRING} 'bash -s' < finished.sh`
num_dirs=$(echo $finished_dirs | wc -w)
if [ $num_dirs -eq 0 ]; then
  exit;
fi
echo $finished_dirs | tr ' ' '\n'
ssh $SSH_STRING "bash -s" -- < hydra_cleanup.sh "$finished_dirs"

