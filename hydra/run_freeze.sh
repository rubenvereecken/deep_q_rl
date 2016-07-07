for freeze in -1 100 1000 2500 5000 10000
do
	qsub -l walltime=300:00:00 -V hydra/run_nips.sh -v ROM=space_invaders,NETWORK_TYPE=nips_cpu -N freeze -o $WORKDIR/out-freeze-${freeze}.txt -F "--freeze-interval=${freeze}"
done
