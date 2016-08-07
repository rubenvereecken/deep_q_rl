#!/bin/sh

LOG_DIR=${LOG_DIR:-"../results"}
HYDRA_WORKDIR="/gpfs/work/rvereeck"
SSH_STRING="rvereeck@hydra.vub.ac.be"

finished_dirs=`ssh rvereeck@hydra.vub.ac.be 'bash -s' < finished.sh`
echo $finished_dirs | tr ' ' '\n'
dirs=""

# Format directories for rsync use
for dir in $finished_dirs; do
  dirs="$dirs:${dir%/} "
done

mkdir -p $LOG_DIR
cd $LOG_DIR
rsync -avz -e ssh $SSH_STRING$dirs . --progress --include "*network_file_[12]00.pkl" --exclude "*network_file*" --exclude "state"
# rsync -avz -e ssh $SSH_STRING$dirs . --progress --exclude "*network_file*" --exclude "state"
