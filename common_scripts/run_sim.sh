#!/bin/bash
#SBATCH -J JOBNAME
#SBATCH -e error.%A
#SBATCH -o output.%A
#SBATCH -N NUM_NODES
#SBATCH --ntasks-per-node=NUM_PROCESS
#SBATCH --cpus-per-task=1

# Run FEPX in parallel on 48 processors
mpirun -np NUM_PROCESS fepx

exit 0