#!/bin/bash
REPS=${REPS:-5}
for i in {1..5}
do
  echo $i
  export REP=$i
  $@
done
