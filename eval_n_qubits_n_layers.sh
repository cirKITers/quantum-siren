#!/bin/bash
# 
# name of the job for better recognizing it in the queue overview
#SBATCH --job-name=quantum-siren
# 
# define how many nodes we need
#SBATCH --nodes=1
#
# we only need on 1 cpu at a time
#SBATCH --ntasks=1
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=20:00:00
# 
# partition the job will run on
#SBATCH --partition single
# 
# expected memory requirements
#SBATCH --mem=8000MB
#
# output path
#SBATCH --output="logs/slurm/slurm-%j.out"

for n_qubits in 2,4,6,8
do
    for n_layers in 5,10,20,30
    do
        ./slurm_job.sh training.n_qubits:$n_qubits,training.n_layers:$n_layers
    done
done
# Done
exit 0



