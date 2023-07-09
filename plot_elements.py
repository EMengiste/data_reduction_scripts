import os
from ezmethods import *
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import multiprocessing
import time
# Latex interpretation for plots
# Latex interpretation for plots
plt.rcParams.update({'font.size': 55})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.figsize'] = 10.62,9.22

# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1

home = "/Users/ezramengiste/Documents/work_stuff/the_sims/"
home="/media/schmid_2tb_1/etmengiste/files/slip_system_study/"
#print(os.listdir(home))
### inni
elts=False
fs=0.6
sty = "solid"
val = 1
leng = 0.05
lw=2
step ="28"
grain_id = "165"
slips,set_num,aniso = ["4",1,400]
##

elset_ids={}
for i in range(1,2001):
       elset_ids[str(i)] = []
file = open("Cube.stelt").readlines()
for i in file:
       splitted=i.split()
       elset_ids[splitted[1]].append(splitted[0])

fig = plt.figure()
ax = fig.add_subplot(projection='3d')

#add get num grains function
#sim = fepx_sim("file",home+"1_uniaxial")
ss = ["2","4","6"]
ani=[125,150,175,200,300,400]
sim_name= (ss.index(slips)*30)+(len(ani)*(set_num-1)+ani.index(aniso)+1)

#
sim_name="000"+str(sim_name)
sim_name = sim_name[-3:]
#
sim = fepx_sim("Cube.sim",path=home+sim_name+"/Cube.sim")
file = open(home+sim_name+"/Cube.sim/inputs/simulation.msh").readlines()
sim.post_process()
num_elts = int(sim.sim["general"].split()[2])

elt_cols = [0 for i in range(num_elts)]

values=[]
for i in file:
       splitted=i.split()
       if splitted[0] == "$Elements":
              print(i)
              elts=True
       elif splitted[0] == "$EndElements":
              print(i)
              elts=False
       if elts:
              if len(splitted)>1 and splitted[1]=="11":
                     values.append(splitted)
print(len(values))


top= [12,13,14,
      19,20,
      22,23,24,
      26,
      31,
      34,
      36,37,38,39,
      43,
      46,47,48,
      56,57,58,59,
      61,
      64,65,66,67,
      69,
      74,75,
      86,87,88,
      91,92,
      95,
      97,98,
      100,
      105,106,107,
      117,
      119,
      124,125,
      128]
bottom= [0,1,2,3,4,5,6,7,8,9,10,11,
         15,16,17,18,
         21,
         25,
         27,28,29,30,
         32,33,
         35,
         40,41,42,
         44,45,
         49,50,51,52,53,54,55,
         60,
         62,63,
         68,
         70,71,72,73,
         76,77,78,79,80,81,82,83,84,85,
         89,90,
         93,94,
         96,
         99,
         101,102,103,104,
         108,109,110,111,112,113,114,115,116,
         118,
         120,121,122,123,
         126,127,
         129,130,131,132]

#pprint(elset_ids[grain_id])
print("made it this far firsty")
elts = elset_ids[grain_id]
#exit(0)
cols = ["k","k","k"]
plot_stress_triad = False
for ind,val in enumerate(elts):
       #
       # Varify the indexing between internal ids and neper nums
       vals= values[int(val)-1]
       #print(vals)
       vals= [int(i)-1 for i in vals[6:]]
       #print(vals)
       #exit(0)
       nodes = np.array(sim.get_output("coo",ids=vals,step=step,res="nodes")).T
       color = "k"
       if ind in top:
              color="r"
              elt_cols[int(val)-1] = 1
       if ind in bottom:
              color="b"
              elt_cols[int(val)-1] = 3
       x,y,z = nodes[0], nodes[1], nodes[2]
       ax.scatter(x,y,z,s=20,c=color)
       x,y,z = avg(nodes[0]), avg(nodes[1]), avg(nodes[2])
       start = [x,y,z]
       ax.scatter(x,y,z,s=200,c=color)
       #ax.text(x,y,z,str(elts.index(val)),fontsize=7)
       #
       # try orientation rot mat=
       #ax.scatter(nodes[0],nodes[1],nodes[2],s=3)
       #axis_basis = R.from_mrp(np.array(oris)).as_matrix().T
       #axis_basis = [oris]
       #
       cols = ["k","k","k"]
       # try plotting stress triad
       if plot_stress_triad:
              stress = sim.get_output("stress",step=step,res="elts",id=val)
              stress_mat= to_matrix(np.array(stress))
              eig_val, eig_vect = np.linalg.eig(stress_mat)
              norm_eigval=normalize_vector(eig_val)
              ind_max = list(norm_eigval).index(max(norm_eigval))
              cols[ind_max]="r"
              #
              col="k"
              for ind,eig in enumerate(eig_vect):
                     ax.quiver(start[0],start[1],start[2],
                            eig[0],eig[1],eig[2]
                            ,length=norm_eigval[ind]*leng,normalize=True
                     ,color=cols[ind], linestyle = sty,linewidth=lw)
       ###
       ###
       ###
#print(elt_cols)
file = open("element_color_mask","w")
for i in elt_cols:
       file.write(str(i)+"\n")
offset= np.array([0,0,0])
start=np.array([0.2,0.230,0.755])
xyz_offset = [[0,0,0],[0,0,0],[0,0,0]]
coordinate_axis(ax,start,space="real",fs=40,leng=0.04,offset_text=1.3,offset=offset,xyz_offset=xyz_offset)
#
#ele,azi,roll =[31,-28,0]
ele,azi,roll =[45,45,0]
ax.view_init(elev=ele, azim=azi)
ax.set_aspect("equal")
ax.axis("off")
plt.grid(False)
show = True
show = False
if show:
       plt.show()
else:
       #fig.savefig("funda_region")
       fig.savefig("bad_elt")
       #fig.savefig("funda_region_zoomed_grain_id_"+str(grain_id))
exit(0)
#####################################################
#####################################################
#