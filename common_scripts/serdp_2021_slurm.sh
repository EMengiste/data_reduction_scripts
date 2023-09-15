#!/bin/bash
#SBATCH -J JOBNAME
#SBATCH -e error.%A
#SBATCH -o output.%A
#SBATCH -N 1
#SBATCH --ntasks=40
#SBATCH --cpus-per-task=1


# If you run on one of the workstations, use:
mpirun -np 40 /home/etmengiste/code/FEPX-dev/build/fepx

# If you run on the HPC, use:
#srun --mpi=pmi2 fepx

exit 0
