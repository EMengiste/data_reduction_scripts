#!/bin/bash

# for a in $(seq 0 1 $1); do
#       convert -gravity south +append ${a}.png\
#                       output_${a}.png              \
#                       output_${a}_density.png      \
#                       output_${a}_density_pf.png   \
#                       image_ipf_$a.png
#     echo combined_step $a
# done

echo making gif
make_gif(){
convert                                                  \
  -delay 7.7576                                              \
   $(for i in $(seq 0 1 $1); do echo ${2}${i}${3}.png; done) \
  -loop 0                                                \
   ../animated${2}${3}_ipf.gif ;
}
make_gif  66 output_ 
make_gif  66 output_ _pf
make_gif  66 output_ _density
make_gif  66 output_ _density_pf
make_gif  66 odf_int_
make_gif  66 odf-rod_space_
# make_gif  66 image_ipf_
exit 0

# This script produces the files:
# - post.report
# - post.coo.core*
# - post.strain-el.core*
# - post.stress.core*
# - post.force.*
