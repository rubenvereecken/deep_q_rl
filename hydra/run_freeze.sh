for freeze in -1 100 1000 2500 5000 10000
do
	qsub -l walltime=300:00:00 -V hydra/run.sh -v '' -N freeze-${freeze} -F "--freeze-interval=${freeze}"
done
