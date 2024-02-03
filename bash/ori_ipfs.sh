#!/bin/bash
#

rad=0.01

cu=${PWD##*/}_$1
neper -V ../simulation.sim \
      -simstep $1\
      -datanodecoofact 4\
      -datanodecoo coo           \
      -dataelt1drad 0.004        \
      -dataelt3dedgerad 0.000015   \
      -dataelt3dcol stress33     \
      -dataeltscaletitle "Stress-zz (MPa)" \
      -dataeltscale 0:270        \
      -showelt1d all             \
      -imagesize 700:920         \
      -showcsys 0\
      -cameraangle 12            \
      -print defromed_$1 \

echo plotting deformed mesh

# convert -gravity south +append \
#         defromed_$1.png              \
#         scalebar-scale3d.png              \
#         $1.png                 \
#         motion_pictures_$1.png \
#         image_$1.png

echo  stitching image_$1.png

neper -V "initial(type=ori):file(ini,des=rodrigues:active),path(type=ori):file(ori_$1,des=rodrigues:active)" \
        -space ipf                                                          \
        -imagesize 1000:1000                                                \
        -datainitialrad 0.030 -datainitialcol red  -datainitialedgecol red  \
        -datapathrad $rad  -datapathcol black                         \
        -print ${cu}\
        -space pf                                                           \
        -pfpole 1:0:0                                                       \
        -datainitialrad 0.030 -datainitialcol red  -datainitialedgecol red  \
        -datapathrad $rad  -datapathcol black                         \
        -print ${cu}_pf
              
convert ${cu}.png -fill white -draw "rectangle 0,0 380,200" ${cu}.png  \


echo plotting ipf

neper --rcfile none  -V ../simulation.sim\
      -simstep $1 -space ipf -ipfmode density             \
      -dataelsetscale 0:2.5\
      -print ${cu}_density\
      -space pf -pfmode density    \
      -dataelsetscale 0:4\
      -pfpole 1:0:0                 \
      -print ${cu}_density_pf

#### make rodrigues space heatmap
neper --rcfile none -V ../simulation.sim/orispace/fr-cub.msh\
        -datanodecol "real:file(../simulation.sim/results/mesh/odfn/odfn.step$1)" \
        -datanodescaletitle "MRD" \
        -dataeltcol from_nodes \
        -datanodescale 0.0:0.5:1.0:1.5:2.0:2.5:3.0:3.5 \
        -showelt1d all \
        -showcsys 0 \
        -dataelt1drad 0.002 \
        -dataelt3dedgerad 0 \
        -cameracoo 4:4:3 -cameraprojection orthographic -cameraangle 13\
        -imagesize 1000:1000 \
        -print odf-rod_space_$1\
        -slicemesh "x=0,y=0,z=0" \
        -showmesh 0 \
        -imagesize 1000:1000 \
        -datanodescaletitle "MRD" \
        -print odf_int_$1

echo plotting ipf
# neper --rcfile none  -V fr-cub.tess,'pts(type=ori):file(ini)',"step(type=ori):file(ori_$1)" -datacellcol lightblue \
#         -datacelltrs 0.5 -dataedgerad 0.003 \
#         -dataptscol red -dataptsrad 0.030\
#         -datastepcol black -datasteprad 0.01\
#         -cameracoo 4:4:3 -cameraprojection orthographic -cameraangle 13\
#         -imagesize 1000:1000 -print fr-cub

convert +append odf-rod_space_$1.png odf-rod_space_$1-scalen.png odf-rod_space_$1.png
convert +append odf_int_$1.png odf_int_$1-scalen.png odf_int_$1.png
rm odf-rod_space_$1-scalen.png
rm odf_int_$1-scalen.png
exit 0