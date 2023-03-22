#!/bin/bash
#
# colors
one=green
two=blue
three=yellow
four=orange
five=red
# number of slip systems and set number
slips=("2" "4" "6")
sets=("1" "2" "3" "4" "5")

cu_cont="Cube_control"
el_cont="Elongated_control"
for ss in ${slips[*]};do
 	for set in ${sets[*]};do
 		 # Cubic Domain orientations
 		 cu1="Cube_125_${ss}ss_set_${set}"
 		 cu2="Cube_150_${ss}ss_set_${set}"
 		 cu3="Cube_175_${ss}ss_set_${set}"
 		 cu4="Cube_200_${ss}ss_set_${set}"
 		 cu5="Cube_400_${ss}ss_set_${set}"

  		# Elongated Domain orientations
  		el1="Elongated_125_${ss}ss_set_${set}"
  		el2="Elongated_150_${ss}ss_set_${set}"
  		el3="Elongated_175_${ss}ss_set_${set}"
  		el4="Elongated_200_${ss}ss_set_${set}"
  		el5="Elongated_400_${ss}ss_set_${set}"

  		neper -V "initial(type=ori):file(ini),Control(type=ori):file(${cu_cont}),One(type=ori):file(${cu1}),Two(type=ori):file(${cu2}),Three(type=ori):file(${cu3}),Four(type=ori):file(${cu4}),Five(type=ori):file(${cu5})" \
  		 	 -space ipf                                     \
			   -datainitialrad 0.020 -datainitialcol white    \
         -dataControlrad 0.007 -dataControlcol black    \
  			 -dataOnerad 0.007     -dataOneedgerad 0.0001   -dataOnecol $one\
     		 -dataTworad 0.007     -dataTwoedgerad 0.0001   -dataTwocol $two\
			   -dataThreerad 0.007   -dataThreeedgerad 0.0001 -dataThreecol $three\
			   -dataFourrad 0.007    -dataFouredgerad 0.0001  -dataFourcol $four\
			   -dataFiverad 0.007    -dataFiveedgerad 0.0001  -dataFivecol $five\
			   -print Cube_set_${set}_${ss}ss

 		 neper -V "initial(type=ori):file(ini),Control(type=ori):file(${el_cont}),One(type=ori):file(${el1}),Two(type=ori):file(${el2}),Three(type=ori):file(${el3}),Four(type=ori):file(${el4}),Five(type=ori):file(${el5})" \
    		-space ipf                                     \
   		  -datainitialrad 0.020 -datainitialcol white    \
        -dataControlrad 0.007 -dataControlcol black    \
    		-dataOnerad 0.007     -dataOneedgerad 0.0001   -dataOnecol $one\
    		-dataTworad 0.007     -dataTwoedgerad 0.0001   -dataTwocol $two\
   			-dataThreerad 0.007   -dataThreeedgerad 0.0001 -dataThreecol $three\
   			-dataFourrad 0.007    -dataFouredgerad 0.0001  -dataFourcol $four\
    		-dataFiverad 0.007  -dataFiveedgerad 0.0001 -dataFivecol $five\
   			-print Elongated_set_${set}_${ss}ss
	done
done
exit 0
