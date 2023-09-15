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
#! /bin/bash
#SBATCH -p main
#SBATCH -J SlurmJobName
#SBATCH -e error.%A
#SBATCH -o output.%A
#SBATCH --nodes=NUM_NODES
#SBATCH --ntasks-per-node=64
#SBATCH --cpus-per-task=1
#SBATCH --time=8:00:00

# Load mambaforge if needed
. ~/.bashrc
mamba activate work

# Load modules if needed
module load vasp/6.3.2
module load vasp/potpaw54

srun --mpi=pmi2 vasp_std
