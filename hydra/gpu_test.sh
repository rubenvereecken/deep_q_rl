#!/bin/bash -l
#PBS -l nodes=1:ppn=1:gpus=1:gpgpu
#PBS -l mem=1gb
#PBS -l walltime=1:00:00
#PBS -m e
#PBS -d /u/rvereeck/deep_q_rl/deep_q_rl/
#PBS -o /gpfs/work/rvereeck/test-gpu.txt
#PBS -j oe

module load CUDA/7.5.18
THEANO_FLAGS=device=gpu python -c "import theano; print(theano.sandbox.cuda.device_properties(0))"
