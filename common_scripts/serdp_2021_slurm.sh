#!/bin/bash
#SBATCH -J JOBNAME
#SBATCH -e error.%A
#SBATCH -o output.%A
#SBATCH -N 1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1


# If you run on one of the workstations, use:
# /home/etmengiste/code/FEPX-dev/build/
mpirun -np 48 fepx

# If you run on the HPC, use:
#srun --mpi=pmi2 fepx

exit 0
