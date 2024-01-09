#!/bin/bash 
#SBATCH -J job_name
#SBATCH -e error.%A 
#SBATCH -o output.%A 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1

python3 serdp_main.py

exit 0
