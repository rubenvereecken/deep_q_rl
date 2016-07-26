for game in $(dir "roms")
do
  export ROM=${game%%.*}
  echo "Starting up $ROM"
  qsub hydra/run_cpu.sh $@ -V -N "$PREFIX-$ROM"
done
