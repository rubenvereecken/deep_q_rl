REPS=5
for i in {1..$REPS}
do
  export REP=$i
  $@
done
