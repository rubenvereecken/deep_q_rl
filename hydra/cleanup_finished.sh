#!/bin/bash
SSH_STRING="rvereeck@hydra.vub.ac.be"

finished_dirs=`ssh ${SSH_STRING} 'bash -s' < finished.sh`
echo $finished_dirs | tr ' ' '\n'
ssh $SSH_STRING "rm -rf $finished_dirs"
