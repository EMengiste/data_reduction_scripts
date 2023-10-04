#!/bin/bash 
#SBATCH -J job_name
#SBATCH -e error.%A 
#SBATCH -o output.%A 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
mpirun -np 48 fepx
exit 0