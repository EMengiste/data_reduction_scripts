#!/bin/bash
#
# sampled grains
start=500
sample_num=100
# sets
ss=2
set=5
# colors
one=green
two=blue
three=yellow
four=orange
five=red
#cube ori files
cu1="Cube_125_${ss}ss_set_${set}"
cu2="Cube_150_${ss}ss_set_${set}"
cu3="Cube_175_${ss}ss_set_${set}"
cu4="Cube_200_${ss}ss_set_${set}"
cu5="Cube_400_${ss}ss_set_${set}"
#elongated ori files
el1="Elongated_125_${ss}ss_set_${set}"
el2="Elongated_150_${ss}ss_set_${set}"
el3="Elongated_175_${ss}ss_set_${set}"
el4="Elongated_200_${ss}ss_set_${set}"
el5="Elongated_400_${ss}ss_set_${set}"

neper -V "initial(type=ori):file(ini),One(type=ori):file(${cu1}),Two(type=ori):file(${cu2}),Three(type=ori):file(${cu3}),Four(type=ori):file(${cu4}),Five(type=ori):file(${cu5})" \
 	   -space ipf                                     \
     -datainitialrad 0.020 -datainitialcol white    \
	   -dataOnerad 0.007     -dataOneedgerad 0.0001   -dataOnecol $one\
   	 -dataTworad 0.007     -dataTwoedgerad 0.0001   -dataTwocol $two\
  	 -dataThreerad 0.007   -dataThreeedgerad 0.0001 -dataThreecol $three\
     -dataFourrad 0.007    -dataFouredgerad 0.0001  -dataFourcol $four\
     -dataFiverad 0.007    -dataFiveedgerad 0.0001  -dataFivecol $five\
     -print Cube_set_${set}_${ss}ss

neper -V "initial(type=ori):file(ini),One(type=ori):file(${el1}),Two(type=ori):file(${el2}),Three(type=ori):file(${el3}),Four(type=ori):file(${el4}),Five(type=ori):file${el5}" \
      -space ipf                                     \
      -datainitialrad 0.020 -datainitialcol white    \
	    -dataOnerad 0.007     -dataOneedgerad 0.0001   -dataOnecol $one\
      -dataTworad 0.007     -dataTwoedgerad 0.0001   -dataTwocol $two\
      -dataThreerad 0.007   -dataThreeedgerad 0.0001 -dataThreecol $three\
      -dataFourrad 0.007    -dataFouredgerad 0.0001  -dataFourcol $four\
      -dataFiverad 0.007  -dataFiveedgerad 0.0001 -dataFivecol $five\
      -print Elongated_set_${set}_${ss}ss
exit 0
