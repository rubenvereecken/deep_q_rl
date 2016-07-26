#!/bin/bash

for pid in `./hydra/pids_running.sh`
do
  out=`checkjob $pid 2>&1`
  echo $out | grep -o '.*gpu.*'
  #echo $out 2>&1 | grep -o '.*AName.*'
  #echo $out | grep -o '.*Features.*'
  #echo $out | grep -o '.*Allocated.*'
  #echo $out | grep -o '.*Reservation.*' 
done
