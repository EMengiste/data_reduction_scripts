#!/bin/bash
#

data_path="/media/schmid_2tb_1/etmengiste/files/slip_system_study/isotropic/Cube.sim"

title="Stress-yy (MPa)"
facts=(3 4)

####
for fact_val in ${facts[@]} ; do
      # neper -V $data_path \
      #       -loop STEP 0 1 27\
      #       -simstep STEP\
      #       -datanodecoofact $fact_val\
      #       -datanodecoo coo           \
      #       -dataelt1drad 0.004        \
      #       -dataelt3dedgerad 0.000015   \
      #       -dataelt3dcol stress33     \
      #       -dataeltscaletitle "Stress-yy (MPa)" \
      #       -dataeltscale 0:270        \
      #       -showelt1d all             \
      #       -imagesize 700:920         \
      #       -showcsys 0\
      #       -cameraangle 12            \
      #       -print 1_${fact_val}_STEP \
      #       -endloop
            
      for a in {0..27}; do
            # # convert value$a.png -transparent white value$a.png
            # neper -V $data_path \
            #       -simstep ${a}\
            #       -datanodecoofact $fact_val\
            #       -datanodecoo coo           \
            #       -dataelt1drad 0.004        \
            #       -dataelt3dedgerad 0.000015   \
            #       -dataelt3dcol stress33     \
            #       -dataeltscaletitle "Stress-yy (MPa)" \
            #       -dataeltscale 0:270        \
            #       -showelt1d all             \
            #       -imagesize 700:920         \
            #       -showcsys 0\
            #       -cameraangle 12            \
            #       -print 1_${fact_val}_${a} \

            convert -gravity south +append motion_pictures_${a}.png \
                        ${a}.png                 \
                        1_${fact_val}_$a.png              \
                        scalebar-scale3d.png              \
                        image_$a.png

            echo  stitching image_$a.png
      done

      convert                                                       \
            -delay 5                                                \
            $(for i in $(seq 0 1 4); do echo image_${i}.png; done)  \
            -loop 0                                                 \
            -delay 20                                               \
            $(for i in $(seq 5 1 27); do echo image_${i}.png; done) \
            -loop 0                                                 \
            real_time_${fact_val}_def.gif

      echo real_time_${fact_val}_def.gif done
done

###########
########333


exit 0

convert                                                  \
  -delay 30                                             \
   $(for i in $(seq 0 1 27); do echo image_${i}.png; done) \
  -loop 0                                                \
   animated30.gif

convert                                                  \
  -delay 40                                             \
   $(for i in $(seq 0 1 27); do echo image_${i}.png; done) \
  -loop 0                                                \
   animated40.gif

exit 0
neper -V $data_path \
      -loop STEP 0 1 27\
      -simstep STEP\
      -datanodecoo coo           \
      -dataelt1drad 0.004        \
      -dataelt3dedgerad 0.000015   \
      -dataelt3dcol ori     \
      -dataelt3dcolscheme ipf     \
      -showelt1d all             \
      -imagesize 700:920         \
      -showcsys 0\
      -cameraangle 12            \
      -print value_ipfSTEP \
      -endloop

convert                                                  \
  -delay 40                                              \
   $(for i in $(seq 0 1 27); do echo image_ipf_${i}.png; done) \
  -loop 0                                                \
   animated_ipf.gif ;

for a in {0..27}; do
      convert -gravity south +append motion_pictures_${a}.png \
                      ${a}.png                 \
                      value_ipf$a.png              \
                      image_ipf_$a.png
done

exit 0
