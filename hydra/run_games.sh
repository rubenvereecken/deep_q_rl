for game in $(dir "../roms")
do
  export ROM=${game%%.*}
  echo "Starting up $ROM"
  qsub run_nips.sh $@ -o "$WORKDIR/out-$ROM.txt"
done
