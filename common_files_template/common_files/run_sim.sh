#!/bin/bash
#SBATCH -J JOBNAME
#SBATCH -e error.%A
#SBATCH -o output.%A
#SBATCH -N 1
#SBATCH --ntasks=48
#SBATCH --cpus-per-task=1

# Run the `Uniaxial Control' example (`1_uniaxial')
# - Executes FEPX with OpenMPI's `mpirun' on 4 processors by default.

# Run FEPX in parallel on 48 processors
mpirun -np 48 ~/code/fepx_multi_g_0_input/fepx/build/fepx

exit 0

# This script produces the files:
# - post.report
# - post.coo.core*
# - post.strain-el.core*
# - post.stress.core*
# - post.force.*
