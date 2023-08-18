#!/bin/bash
# runs the clean up script and then releases the jobs that are held

# make so it changes the jobs released here  using sed
sbatch --job-name=clnupisl1 --hint=nomultithread /media/schmid_1tb_2/etmengiste/aps_add_slip/common_files/post-process.sh


sbatch --job-name=clnupisl2 --hint=nomultithread /home/etmengiste/code/data_reduction_scripts/clean.sh
