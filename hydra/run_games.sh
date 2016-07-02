for game in $(dir "../roms")
do
  ROM=${game%%.*}
  ./run_nips.sh $@
done
