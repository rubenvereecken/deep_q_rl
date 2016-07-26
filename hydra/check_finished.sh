#!/bin/sh

finished_dirs=`ssh rvereeck@hydra.vub.ac.be 'bash -s' < finished.sh`
echo $finished_dirs | tr ' ' '\n'
