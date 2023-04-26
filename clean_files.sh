#!/bin/bash
#SBATCH -J JOBNAME
#SBATCH -e error.%A
#SBATCH -o output.%A
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Run the `Uniaxial Control' example (`1_uniaxial')
# - Executes FEPX with OpenMPI's `mpirun' on 4 processors by default.

# Run FEPX in parallel on 48 processors
python3 /home/etmengiste/code/data_reduction_scripts/clean.py

exit 0