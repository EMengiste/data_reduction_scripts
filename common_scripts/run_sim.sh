#!/bin/bash
#SBATCH -J JOBNAME
#SBATCH -e error.%A
#SBATCH -o output.%A
#SBATCH --nodes=NUM_NODES
#SBATCH --ntasks-per-node=NUM_PROCESS

# Run FEPX in parallel on 48 processors
mpirun -np NUM_PROCESS fepx

exit 0