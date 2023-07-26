import sys
import math
sys.path.append("/Users/ezramengiste/Documents/work_stuff/data_reduction_scripts")
from ezmethods import fepx_sim,pprint,avg,normalize_vector,rot_mat,dif_degs

def rod_to_angle_axis(rod):
    axis,mag = normalize_vector(rod,magnitude=True)
    angle = 2*math.atan(mag)
    return [angle,axis]

def angle_axis_to_mat(angle,axis):
    cos_thet = math.cos(angle)
    sin_thet = math.sin(angle)
    u_x,u_y,u_z = axis
    r11 = cos_thet + (u_x ** 2 * (1-cos_thet))
    r12 = (u_x * u_y * (1-cos_thet)) - (u_z * sin_thet)
    r13 = (u_x * u_z * (1-cos_thet)) + (u_y * sin_thet)

    r21 = (u_y * u_x * (1-cos_thet)) + (u_z * sin_thet)
    r22 = cos_thet + (u_y ** 2 * (1-cos_thet))
    r23 = (u_y * u_z * (1-cos_thet)) - (u_x * sin_thet)

    r31 = (u_z * u_x * (1-cos_thet)) - (u_y * sin_thet)
    r32 = (u_z * u_y * (1-cos_thet)) + (u_x * sin_thet)
    r33 = cos_thet + (u_z ** 2 * (1-cos_thet))
    Rot_Mat = [ [r11,r12,r13],
                [r21,r22,r23],
                [r31,r32,r33]]
    return Rot_Mat


def match_grain(set_1,set_2,ang_res=0.001,dist_res=0.001,debug=False):
    set_1_coo,set_1_oris= set_1
    set_2_coo,set_2_oris= set_2
    if debug:
        print("Coos_1")
        pprint(set_1_coo)
        print("oris_1")
        pprint(set_1_oris)
        print("Coos_2")
        pprint(set_2_coo)
        print("oris_2")
        pprint(set_2_oris)
        print(set_1_oris[0])
        print(set_2_oris[0])
    #
    for i in range(len(set_1_oris)):
        ori_1 = set_1_oris[i]
        ori_2 = set_2_oris[i]
        #
        ori_1 = rod_to_angle_axis(ori_1)
        ori_2 = rod_to_angle_axis(ori_2)
        ori_1 = angle_axis_to_mat(ori_1[0], ori_1[1])
        ori_2 = angle_axis_to_mat(ori_2[0], ori_2[1])
        print(dif_degs(ori_1,ori_2))



sim_path = "/home/etmengiste/jobs/hexa_try/03_hcp.sim"
sim=fepx_sim("name",path=sim_path)
file = open(sim_path+"/inputs/simulation.msh").readlines()
sim.post_process()
num_elts = int(sim.sim["general"].split()[2])
debug=False
elts=False
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
                     values.append(splitted[6:])

if debug:
       pprint(values,max=80)

step_vals = []
for step in range(0,3):
	elt_centroid = []
	elt_ori = []
	for elt_id in range(0,10):
		ori = sim.get_output("ori",res="elts",step=str(step),ids=[elt_id])
		elt_ori.append(ori)
		elt_nodes = values[elt_id]
		X,Y,Z = [],[],[]
		for node in elt_nodes:			
			x,y,z = sim.get_output("coo",res="nodes",ids=[int(node)])
			X.append(x)
			Y.append(y)
			Z.append(z)
		elt_centroid.append([avg(X),avg(Y),avg(Z)])
	step_vals.append([elt_centroid,elt_ori])
print(len(step_vals))

print("step0-1")
match_grain(step_vals[0],step_vals[1])
print("step1-2")
match_grain(step_vals[1],step_vals[2])
exit(0)
####
grain_elt = {}
elt_grain = open("simulation.stelt").readlines()
for i in elt_grain[:3]:
    elt,grain = i.split()
    try:
        grain_elt[grain].append(elt)
    except:
        grain_elt[grain] =[]

print(grain_elt)