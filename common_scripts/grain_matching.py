import sys
import math
# sys.path.append("/home/etmengiste/code/data_reduction_scripts/common_scripts")
from fepx_sim import *
from plotting_tools import *
from tool_box import *

import time
import multiprocessing 
import os
import numpy as np

def avg(arr):
    return sum(arr)/len(arr)

def matrix_to_angle_axis(mat):
    r11,r21,r31= mat[0]
    r12,r22,r32= mat[1]
    r13,r23,r33= mat[2]
    angle = acos((r11+r22+r33 -1)/2)
    x = (r23-r32)/((r23-r32)**2+(r13-r31)**2+(r21-r12)**2)**0.5
    y = (r13-r31)/((r23-r32)**2+(r13-r31)**2+(r21-r12)**2)**0.5
    z = (r21-r12)/((r23-r32)**2+(r13-r31)**2+(r21-r12)**2)**0.5
    return [angle,[x,y,z]]


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
        ori_1 =rod_to_quat(set_1_oris[i])
        ori_2 =rod_to_quat(set_2_oris[i])
        #
        print(ori_1)
        print(ori_2)
        print(degrees(acos(ori_1[0]))*2)
        print(degrees(acos(ori_2[0]))*2)
        prod =quat_misori(ori_2,ori_1)
        print(prod)
        exit(0)
        print(ori_2)
        misori =degrees(acos(prod[0]))*2
        print(misori)
        exit(0)


#
def test_misori():
    rot1 = np.array([[-0.490727000000000,0.675748999999999,-0.550046000000000],
                     [-0.745040000000000,-0.652751000000000,-0.137231999999999],
                     [-0.451777000000000,0.342462000000000,0.823782000000000]]) # L1 - Grain ID- 25876, my ID 15551002962

    rot2 = np.array([[-0.490741000000000,-0.675779999999999,0.549995000000000],
                 [-0.745053000000000,0.652735000000000,0.137231999999999],
                 [-0.451739000000000,-0.342430000000000,-0.823816000000000]]) # Grain ID 23912 in L2 ,newID -1552002911
    order = [1-i for i in range(3)]
    print(order)
    rot1 = rot1[order]
    rot1 =matrix_to_angle_axis(rot1)
    rot1 =quat_of_angle_ax(rot1[0],rot1[1])
    print(normalize_vector(rot1))
    rot2 =matrix_to_angle_axis(rot2)
    rot2 =quat_of_angle_ax(rot2[0],rot2[1])
    #print(normalize_vector(rot2))
    exit(0)
    angle = pi
    raxis=[1,1,1]
    # angle = pi/2
    # raxis=[1,1,1]
    a1 = quat_of_angle_ax(angle, raxis)
    a2 = quat_of_angle_ax(angle, raxis)
    print(a1)
    a1 = quat_to_angle_axis(a1)
    print(a1)


    a1=normalize_vector([.1,.2,.3,.4])
    a2=normalize_vector([.5,.6,.7,.8])
    prod = quat_prod(a1,a2)
    print(prod)
    # misori = normalize_vector([-28,4,6,8])
    misori =degrees(acos(misori[0]))*2

    print(misori)
    exit(0)

#
#
if __name__=="__main__":

    test_misori()
    exit(0)
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