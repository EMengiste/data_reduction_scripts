#------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
import numpy as np
import math
import pandas as pd
import os

def angle_axis_to_mat(angle,axis,scale=1):
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


def vect_to_azim_elev(vect):
    x,y,z = vect
    mag_tot = (x**2 +y**2 +z**2)**0.5
    mag_xy = (x**2 +y**2)**0.5
    azi = math.degrees(math.asin(y/mag_xy))
    ele = math.degrees(math.atan(z/mag_xy))
    return [ele,azi]

def quat_to_angle_axis(quat):
    angle = 2* math.acos(quat[0])
    axis = scalar_multi(quat[1:],1/math.sin(angle/2))
    return angle,axis

def find_nearest(a, a0):
    # https://stackoverflow.com/a/2566508
    "Element in nd array `a` closest to the scalar value `a0`"
    a=np.array(a)
    idx = np.abs(a - a0).argmin()
    return idx
#
## yield calculation for simulation data
def find_yield(stress, strain, offset="",number=""):
    load_steps= len(strain)
    #
    if offset == "":
        offset = 0.002
    if number == '':
        E = stress[1]/ strain[1]
        index = ''
        stress_off = [(E * i) - (E * offset) for i in strain]
        for i in range(load_steps):
            if  stress_off[i] > stress[i]:
                x_n1 = i
                x_n = x_n1 - 1
                break
        # Use Cramer's rule to find where lines intersect (assume linear between
        #  simulation points)
        m1 = E
        a = - E *  offset
        #
        m2 = (stress[x_n1] - stress[x_n]) / (strain[x_n1] - strain[x_n])
        b = stress[x_n] - (m2 * strain[x_n])
        #
        #
        Ystrain = (a - b) / (m2 - m1)
        Ystress = ((a * m2) - (b * m1)) / (m2 - m1)
    else:
        index= find_nearest(stress,number)
        x=np.array(strain[0:index]).reshape((-1, 1))
        y=np.array(stress[0:index])
        model= LinearRegression()
        model.fit(x,y)
        E = model.coef_[0]
        stress_off = [(E * strain) - (E * offset) for strain in strain]
        print('E=', model.coef_)
        print("b=",model.intercept_)
        print('index=',index)
        print('near=',find_nearest(stress,number))
        diff= np.array(stress)-np.array(stress_off)
        x_n1= np.abs(diff).argmin()
        Ystrain = float(strain[x_n1])
        Ystress = float(stress[x_n1])
    values ={"y_stress": Ystress,
             "y_strain": Ystrain,
             "stress_offset": stress_off,
             "index": x_n1}
    return values
  #
 #
#def generate_tess(n,destination_name,source_dir,options={"mode" :"run"}):
    print("\n===")
    tesselation= main_dir+"/input_data"+destination_name

    commands= ["-n "+str(n),"-o "+tesselation]
    neper_command=options["source code"]+" -T "+commands[0]
    # populate the commands using the optional input dictionary
    for i in options:
        if i[0]== "-":
            neper_command+=" "+i+' '+options[i]
            commands.append(i+' '+options[i])
    #
    neper_command+=" "+commands[1]
    if options["mode"] =="debug":
        #print(source_dir)
        print(os.getcwd())
        pprint(commands)
        print("Tess command:\n",neper_command)
    elif options["mode"] =="run":
        print("Tess command:\n",neper_command)
        if os.path.exists(tesselation+".tess"):
            print("tesselation already exists")
        else:
            print("tesselation doesn't exist generating new")
            os.system(neper_command+' '+commands[1])

    #print(tess_destination+".tess")
    return tesselation+".tess"
#
def generate_tess(n,name,main_dir=".",source_code="neper",options={"mode" :"run"}):
    print("\n===")
    curr_dir = os.getcwd()
    os.chdir(main_dir)
    tesselation= name

    commands= ["-n "+str(n),"-o "+tesselation]
    neper_command=source_code+" -T "+commands[0]
    # populate the commands using the optional input dictionary
    for i in options:
        if i[0]== "-":
            neper_command+=" "+i+' '+options[i]
            commands.append(i+' '+options[i])
    #
    neper_command+=" "+commands[1]
    if options["mode"] =="debug":
        #print(source_dir)
        print("-----debugging--------")
        print("Tess command:\n",neper_command)
        print("Commands list:")
        pprint(commands)
        print(os.getcwd())
    elif options["mode"] =="run":
        print("-----tesselation--------")
        print("Tess command:\n",neper_command)
        if os.path.exists(tesselation+".tess"):
            print("tesselation already exists")
        else:
            print("tesselation doesn't exist generating new")
            os.system(neper_command)

    os.chdir(curr_dir)
    return main_dir+"/"+tesselation+".tess"
#
def generate_msh(source_dir,num_partition,source_code="neper",options={"mode" :"run"}):
    print("\n===")
    curr_dir = os.getcwd()
    name = source_dir.split("/")[:]
    mesh_dir = "/".join(name[:-1])
    os.chdir(mesh_dir)
    tess_name =name[-1]
    commands = ["-part "+str(num_partition)]
    neper_command=source_code+" -M "+tess_name+" "+commands[0]
    mesh_name=tess_name
    # populate the commands using the optional input dictionary
    for i in options:
        if i[0]== "-":
            neper_command+=" "+i+' '+options[i]
            commands.append(i+' '+options[i])
        if i== "-o":
            mesh_name= options[i]

    if options["mode"]=="debug":
        #print(source_dir)
        print("-----debugging--------")
        print("Meshing command:\n",neper_command)
        print("Commands list:")
        pprint(commands)
        print(os.getcwd())
    elif options["mode"]=="run":
        print("-----meshing--------")
        print("Meshing command:\n",neper_command)
        if os.path.exists(mesh_dir+"/"+mesh_name+".msh"):
            print("Mesh already exists",mesh_dir+"/"+mesh_name+".msh")
        else:
            print("Mesh doesn't exist generating new")
            os.system(neper_command+" > "+mesh_name+"_output")

    elif options["mode"]=="stat":
        print("-----getting_stats--------")
        print("Meshing command:\n",neper_command)
        neper_command=source_code+" -M -loadmesh "+mesh_name+".msh"            
        for i in options:
            if i.startswith("-stat"):
                neper_command+=" "+i+' '+options[i]
        print(neper_command)
        os.system(neper_command+" > "+mesh_name+"_output")

    elif options["mode"]=="remesh":
        print("-----remeshing--------")
        print("Meshing command:\n",neper_command)
        os.system(neper_command+" > "+mesh_name+"_output")
    os.chdir(curr_dir)
    return mesh_dir+"/"+mesh_name+".msh"
#
def visualize(source_dir,source_code="neper",options={"mode" :"run"}):
    commands = []
    neper_command=source_code+" -V "+source_dir+" "+commands[0]
    # populate the commands using the optional input dictionary
    for i in options:
        if i[0]== "-":
            neper_command+=" "+i+' '+options[i]
            commands.append(i+' '+options[i])

    if options["mode"]=="debug":
        #print(source_dir)
        print(os.getcwd())
        pprint(commands)
        print("Visualization command:\n",neper_command)
    elif options["mode"]=="run":
        print("Visualization command:\n",neper_command)
        if os.path.exists(source_dir[0:-5]+".msh"):
            print("Image already exists",source_dir[0:-5]+".png")
        else:
            print("Image doesn't exist generating new")
            os.system(neper_command+" > vis_output")
    elif options["mode"]=="remesh":
        os.system(neper_command+" > vis_output")
#
def post_process(sim_path,main_dir=".",options={"source code":"neper"}):
    print("\n===")
    print(sim_path)
    neper_command= options["source code"]+" -S ."
    if os.path.exists(main_dir+"/"+sim_path+".sim"):
        print("Simulation folder exists")
        print(sim_path+".sim")
    else:
        print("Simulation folder doesn't exist generating new")
        print(sim_path)
        if options["mode"]=="debug":
            print("<debug_mode>")
            os.chdir(main_dir+"/"+sim_path)
            print(neper_command)
            #
            print(os.listdir())
            #
        elif options["mode"]=="run":
            if os.path.exists("post.report"):
                print("Running post processing commands")
                os.system(neper_command)
            else:
                print("simulation completed")
#
def run_sim(path_to_msh,path_to_config,sim_path,main_dir=".",options={"source code":"fepx"}):
    print("\n===")
    sim_path_full= main_dir+"/"+sim_path
    print(sim_path_full)
    if os.path.exists(sim_path_full):
        print("Simulation folder exists")
        print(sim_path)
    else:
        print("Simulation folder doesn't exist generating new")
        os.makedirs(sim_path_full)
        print(sim_path)

    if options["mode"]=="debug":
        print("<debug_mode>")
        print(main_dir+"/"+sim_path)
        os.chdir(main_dir+"/"+sim_path)
        print(path_to_msh)
        print(path_to_config)
        #
        print(os.listdir())
        #
    elif options["mode"]=="run":
        if not os.path.exists(main_dir+"/"+sim_path+"/simulation.msh"):
            print("Mesh Shortcut doesn't exist")
            os.symlink(path_to_msh,main_dir+"/"+sim_path+"/simulation.msh")
        if not os.path.exists(main_dir+"/"+sim_path+"/simulation.config"):
            print("Config Shortcut doesn't exists")
            os.symlink(path_to_config,main_dir+"/"+sim_path+"/simulation.config")
        else:
            print("Simulation folder sufficient")
        #
        os.chdir(main_dir+"/"+sim_path)
        print("mpirun "+options["source code"]+" -np "+ str(options["cores"]))
        print(os.getcwd)
        if not os.path.exists("post.report"):
            print("Running simulation")
            os.system("mpirun "+options["source code"]+" -np "+ str(options["cores"]))
        else:
            print("simulation completed")
    elif options["mode"]=="remesh":
        os.system(neper_command)
#
def write_precip_file(frac,rad,name="simulation",res="Elset"):
    ####    Open precip distribution file and write header
    precip_dist_file = open(name+".precip",'w')
    precip_dist_file.write("$"+res+"PrecipDistribution\n")
    num_vals=len(frac)
    precip_dist_file.write(str(num_vals)+"\n")
    lines=""
    for i in range(num_vals):
        lines+=str(i+1)+" "+str(frac[i])+" "+str(abs(rad[i]))+"\n"
    precip_dist_file.write(lines)
    precip_dist_file.write("$End"+res+"PrecipDistribution")
    print(f"wrote file {name}")
#  
def write_crss_file(values,target_dir="",name="simulation",res="Elset"):
    ####    Open crss file and write header
    #   set for isotropic input
    #
    precip_dist_file = open(target_dir+name+".crss",'w')
    precip_dist_file.write("$"+res+"Crss\n")
    num_vals=len(values)
    precip_dist_file.write(str(num_vals)+" 1\n")
    for i in range(num_vals):
        precip_dist_file.write(str(i+1)+" "+str(values[i])+"\n")
    precip_dist_file.write("$End"+res+"Crss")

#
def pprint(arr):
    for i in arr:
        print("+=>",i)
#