#!/bin/bash
#
#
# Create tessellation based on statistical data
#
# Sigma and mean are calculated manualy from the radii file.
neper -T \
    -n 2000 \
    -domain "cube(1,1,1)" \
    -morpho "diameq:lognormal(0.99457395126709,0.39969471320642),1-sphericity:lognormal(0.145,0.03)"\
    -reg 1 \
    -statcell diameq\
    -morphooptistop eps=1e-3\
    -o 2000g_cube

neper -T \
    -n 2000 \
    -domain "cube(4,1,0.5)" \
    -morpho "diameq:lognormal(0.99457395126709,0.39969471320642),1-sphericity:lognormal(0.145,0.03)"\
    -reg 1 \
    -statcell diameq\
    -morphooptistop eps=1e-3\
    -o 2000g_4b1b05
#
neper -M 2000g_cube.tess \
    -order 2 \
    -rcl 2\
    -part 48

neper -M 2000g_4b1b05.tess \
    -order 2 \
    -rcl 2\
    -part 48

neper -V 2000g_cube.tess \
    -print simulation_tess

neper -V 2000g_cube.msh \
    -print simulation_mesh

neper -V 2000g_4b1b05.tess \
    -print simulation_tess

neper -V 2000g_4b1b05.msh \
    -print simulation_mesh

exit 0
