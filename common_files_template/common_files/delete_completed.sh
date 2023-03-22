#!/bin/bash


# This script takes the simulation 'mysim.sim' file and creates images for given
#  simulation steps (here its going from step 0-10 in steps of 1)
#

domains=("Cube" "Elongated")
simulation=("001" "002" "003" "004" "005" "006" "007" "008" "009" "010"
            "011" "012" "013" "014" "015" "016" "017" "018" "019" "020"
            "021" "022" "023" "024" "025" "026" "027" "028" "029" "030"
            "031" "032" "033" "034" "035" "036" "037" "038" "039" "040"
            "041" "042" "043" "044" "045" "046" "047" "048" "049" "050"
            "051" "052" "053" "054" "055" "056" "057" "058" "059" "060"
            "061" "062" "063" "064" "065" "066" "067" "068" "069" "070"
            "071" "072" "073" "074" "075")
cd ..
for (( i=0; i<30; i++));do
  echo $i
  echo ${simulation[$i]}
  pwd
  rm -rf ${simulation[$i]}
done

exit 0



