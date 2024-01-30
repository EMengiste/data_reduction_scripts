#!/bin/bash
#
#
#
#
path_tess="/media/schmid_2tb_1/etmengiste/files/slip_system_study/"
#
path_cs="/home/etmengiste/code/data_reduction_scripts/images"
neper -V ${path_tess}/isotropic/Cube.sim \
        -step 0 \
        -datanodecoo coo \
        -datanodecoofact 7 \
        -imagesize 1100:1000\
        -lightambient 1 -lightsource none\
        -dataelsetcol ori -dataelsetcolscheme ipf\
        -cameraangle 15 -showcsys 0\
        -cameracoo 4:-5:4\
        -print Cube_msh        
convert -gravity south +append Cube_msh.png ~/code/stdtriangle_new.png msh_ipf.png
convert -chop 150x0+950+0 msh_ipf.png Cube_msh.png
convert -gravity south +append ${path_cs}/coo_ax.png \
                                Cube_msh.png \
                                Cube_msh_cs.png
convert -chop 160x0+274+0 Cube_msh_cs.png Cube_msh_cs.png


neper -V ${path_tess}/isotropic/Cube.sim \
        -step 0 \
        -datanodecoo coo \
        -datanodecoofact 7 \
        -imagesize 1100:1000\
        -lightambient 1 -lightsource none\
        -dataelsetcol white\
        -cameraangle 15 -showcsys 0\
        -cameracoo 4:-5:4\
        -print mesh       
convert -gravity south +append mesh.png ~/code/stdtriangle_new.png msh_ipf.png
convert -chop 150x0+950+0 msh_ipf.png mesh.png
convert -gravity south +append ${path_cs}/coo_ax.png \
                                mesh.png \
                                msh_cs.png
convert -chop 160x0+274+0 msh_cs.png msh_cs.png
# exit 0 mesh.png
# neper -V ${path_tess}/common_files/Cube.tess \
#         -imagesize 1100:1000\
#         -lightambient 1 -lightsource none\
#         -datacellcol white\
#         -cameraangle 15 -showcsys 0\
#         -cameracoo 4:-5:4\
#         -print tesselation  


convert -gravity south +append ${path_cs}/coo_ax.png \
                                tesselation.png \
                                tess_cs.png

convert -gravity south +append ${path_cs}/coo_ax.png \
                                mesh.png \
                                msh_cs.png
convert -chop 160x0+274+0 tess_cs.png tess_cs.png
convert -chop 160x0+274+0 msh_cs.png msh_cs.png
exit 0 
########### Aps visualization
steps="28"
inc="7"
start="28"
#grain="-showelset id==497 " 
grain="-showelset id==173||id==611||id==612" 

camera=" -cameralookat -x:y:z"
dirs=("isotropic" "020" "030")
#          -imageformat vtk,png \

for dir in ${dirs[*]};do 
        neper -V ../${dir}/Cube.sim \
                -loop STEP $start $inc $steps \
                -step STEP -datanodecoo coo \
                -dataelttrs 0.5\
                -showelt1d 0\
                ${grain}\
                -dataeltcol ori -print img_${dir:(-3)}_STEP
done

for i in $(seq $start $inc $steps);do 
 convert +append $(for dir in ${dirs[*]};do echo img_${dir:(-3)}_${i}.png; done) img${i}.png 
done

#convert -delay 30 $(for i in $(seq $start $inc $steps);do echo img${i}.png; done) -loop 0 comparison.gif

exit 0
#camera="-cameraangle 7 -cameralookat 0.7:-0.2:0.3 -datacsysfontsize 0.4 -datacsysrad 0.003 -datacsyscoo 0.7:-0.5:0.35"
val1="isotropic"
grain="id==28" 
neper -V ../${val1}/Cube.sim \
        -datanodecoo coo \
        -datanodecoofact 17 \
        ${camera}\
        -step 28 \
        -dataeltcol id \
        -showelt1d 0 \
        -showelset 0 \
        -showelset ${grain}\
        -print img${val1} 

val2="030"
neper -V ../${val2}/Cube.sim \
        -datanodecoo coo \
        -datanodecoofact 17 \
        ${camera}\
        -step 28 \
        -dataeltcol id \
        -showelt1d 0 \
        -showelset 0 \
        -showelset ${grain} \
        -print img${val2} 

convert +append img${val1}.png img${val2}.png img.png
exit 0
convert -delay 20 $(for i in ${steps[@]};do echo img${val}_${i}.png; done) -loop 0 grain${val}_497.gif

rm img${val}_*