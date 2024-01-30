#!/bin/bash
#
#
# visualize the tessellation
#   Grain color: orientation
#   Grain color scheme: ipf
#
neper -V simulation.tess \
    -datacellcol ori           \
    -datacellcolscheme ipf     \
    -imagesize 820:910         \
    -cameraangle 12            \
    -print	tess
# visualize the mesh
#   element color: orientation
#   element color scheme: ipf
#
neper -V simulation.msh  \
    -dataeltcol ori            \
    -dataeltcolscheme ipf      \
    -imagesize 820:910         \
    -cameraangle 12            \
    -showcsys 1                \
    -datacsyscoo 0.2:-.4:0     \
    -datacsyslabel "X:Y:Z"     \
    -datacsysrad 0.005         \
    -datacsyslength 0.2        \
    -print	msh

convert +append tess.png\
                  ../common_files/stdtriangle.png\
                msh.png\
                          poly.png

# Generate  discrete pole figures for
#   poles 100 110 and 111
#
neper -V simulation.tess \
    -space pf                  \
    -pfpole 1:0:0              \
    -datacellcol black         \
    -imagesize ${pf_size}:${pf_size} \
    -print	100

neper -V simulation.tess \
    -space pf                  \
    -pfpole 1:1:0              \
    -datacellcol black         \
    -imagesize ${pf_size}:${pf_size} \
    -print	110

neper -V simulation.tess \
    -space pf                  \
    -pfpole 1:1:1              \
    -datacellcol black         \
    -imagesize ${pf_size}:${pf_size} \
    -print	111
#
# Combine 3 pole plots into one image for the sample
convert +append 100.png 110.png 111.png pf.png
convert -append poly.png pf.png sim_input.png
#
# remove separate pole figure files
rm 1*.png
exit 0
