#!/bin/bash
#
#
# Create tessellation based on statistical data
#
#
neper -T \
    -n 1000 \
    -domain "cube(1,1,1)" \
    -morpho "diameq:lognormal(0.19416069685239897,0.25183387555287233),1-sphericity:lognormal(0.145,0.03)"\
    -reg 1 \
    -morphooptistop "itermax=50000"\
    -statcell diameq\
    -o stat

#

neper -V stat.tess \
    -print stat_tess
