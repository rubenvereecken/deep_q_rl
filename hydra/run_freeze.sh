if [[ ! -z $REP ]]; then
  POSTFIX=-rep_$REP
fi
for freeze in -1 1000 10000
do
	qsub -V hydra/run_cpu.sh -N freeze-${freeze}${POSTFIX} -F "--freeze-interval=${freeze}" -l "nodes=1:ppn=1:ivybridge"
done
