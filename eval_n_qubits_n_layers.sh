#!/bin/bash

for n_qubits in 2 4 8 16; do
    for n_layers in 2 4 8 16 32; do
        sbatch --job-name "qs-q$n_qubits-l$n_layers" ./slurm_job.sh "training.n_qubits:$n_qubits,training.n_layers:$n_layers"
    done
done
# Done
exit 0


