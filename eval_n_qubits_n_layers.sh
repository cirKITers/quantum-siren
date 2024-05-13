#!/bin/bash

for n_qubits in 2 3 4 5 6 7; do
    for n_layers in 2 4 6 8 10 12; do
        sbatch --job-name "qs-q$n_qubits-l$n_layers" ./slurm_job.sh "training.n_qubits=$n_qubits,training.n_layers=$n_layers" "slurm"
    done
done
# Done
exit 0



