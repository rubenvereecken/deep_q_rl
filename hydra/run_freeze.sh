for freeze in -1 1000 10000
do
  export REP=$REP
	qsub -V hydra/run_cpu.sh -N freeze-${freeze} -F "--freeze-interval=${freeze}" -l "nodes=1:ppn=1:ivybridge"
done
