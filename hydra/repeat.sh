#!/bin/bash
REPS=${REPS:-5}
for (( i=1; i<=$REPS; i++ ))
do
  echo $i
  export REP=$i
  $@
done
