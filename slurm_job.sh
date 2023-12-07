#!/bin/bash
# 
# define how many nodes we need
#SBATCH --nodes=1
#
# we only need on 1 cpu at a time
#SBATCH --ntasks=1
#
#SBATCH --cpus-per-task=16
#
# expected duration of the job
#              hh:mm:ss
#SBATCH --time=40:00:00
# 
# partition the job will run on
#SBATCH --partition single
# 
# expected memory requirements
#SBATCH --mem=72000MB
#
# output path
#SBATCH --output="logs/slurm/slurm-%x-%j.out"

/home/kit/scc/lc3267/quantum-siren/.venv/bin/python -m kedro run --params=$1

# Done
exit 0


