#!/bin/bash

WORKDIR="/gpfs/work/rvereeck"
cd $WORKDIR
dirs=`ls -d */`
finished_dirs=""
for dir in $dirs; do
  if [[ ! -a "$dir/state" ]]; then
    continue
  fi
  STATE="$(cat $dir/state)"
  if [ "$STATE" != "RUNNING" ]; then
    finished_dirs="$finished_dirs $WORKDIR/$dir"
  fi
done

echo $finished_dirs
