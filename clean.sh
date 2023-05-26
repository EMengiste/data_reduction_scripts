#!/bin/bash
#SBATCH -J JOBNAME
#SBATCH -e error.%A
#SBATCH -o output.%A
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

# Run the `Uniaxial Control' example (`1_uniaxial')
# - Executes FEPX with OpenMPI's `mpirun' on 4 processors by default.

# Run pyhton cleaning script
python3 /home/etmengiste/code/data_reduction_scripts/post_process.py

scontrol release {3531..3540}
#scontrol release {START..END}
exit 0
