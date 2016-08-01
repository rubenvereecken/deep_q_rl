#!/bin/bash
NUM=${1:-4}
for (( i=1; i<=$NUM; i++ ))
do
  SCRIPT='./run_nature.py' qsub -V hydra/run_gpu.sh -v 'NETWORK_TYPE=nature_cudnn' -N 'queue' 
done
echo "Launched $NUM queue scripts"
