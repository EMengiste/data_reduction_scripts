#------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
import os
import time
import math

def find_nearest(a, a0):
    # https://stackoverflow.com/a/2566508
    "Element in nd array `a` closest to the scalar value `a0`"
    a=np.array(a)
    idx = np.abs(a - a0).argmin()
    return idx
#
def quaternion_misorientation(q1,q2):
       # https://gitlab.tudelft.nl/-/snippets/190
       # Input:
       #       q1 = [w1,x1,y1,z1]
       #       q2 = [w2,x2,y2,z2]
       # Output: 
       #        Theta
       print(len(q1))
       w1,x1,y1,z1 = q1
       w2,x2,y2,z2 = q2
       theta = 2*math.acos(w1*w2 +x1*x2 + y1*y2 + z1*z2)
       return theta
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
    input_name= source_dir.split("/")[:]
    mesh_dir = "/".join(input_name[:-1])
    os.chdir(mesh_dir)
    tess_name =input_name[-1]
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
        neper_command=source_code+" -M -loadmesh "+mesh_name+".msh "+commands[0]
        print("Meshing command:\n",neper_command)
        os.system(neper_command+" > "+mesh_name+"_output")
    os.chdir(curr_dir)
    return mesh_dir+"/"+mesh_name+".msh"
#
def visualize(input_source,source_code="neper",options={"mode" :"run"}):
    commands = []
    input_name= input_source.split("/")[-1]
    neper_command=source_code+" -V "+input_source+" "
    # populate the commands using the optional input dictionary
    for i in options:
        if i[0]== "-":
            neper_command+=" "+i+' '+options[i]
            commands.append(i+' '+options[i])
    neper_command+=" -print "+input_name[:-4]
    if options["mode"]=="debug":
        #print(input_source)
        print(os.getcwd())
        pprint(commands)
        print("Visualization command:\n",neper_command)
    elif options["mode"]=="run":
        print("Visualization command:\n",neper_command)
        if os.path.exists(input_source[0:-4]+".png"):
            print("Image already exists",input_source[0:-4]+".png")
        else:
            print("Image doesn't exist generating new")
            print(os.getcwd())
            os.system(neper_command+" > vis_output"+input_name)
    elif options["mode"]=="rerun":
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
def sort_by_vals(arr,mat):
       arr = np.ndarray.tolist(arr)
       mat = np.ndarray.tolist(mat)
       arr_sorted = sorted(arr)
       arr_sorted.sort()
       mat_sorted = []
       for i in range(len(arr)):
              curr_ind =arr.index(arr_sorted[i])
              #print(curr_ind)
              mat_sorted.append(mat[curr_ind])
       return [arr_sorted,mat_sorted]
#
def pprint(arr):
    for i in arr:
        print("+=>",i)
#
def job_submission_script(path,num_nodes,num_processors,fepx_path="fepx",name="job_name"):
    os.chdir(path)
    file = open("run.sh","w")
    file.writelines("#!/bin/bash \n")
    file.writelines("#SBATCH -J "+str(name)+"\n")
    file.writelines("#SBATCH -e error.%A \n")
    file.writelines("#SBATCH -o output.%A \n")
    file.writelines("#SBATCH --nodes="+str(num_nodes)+"\n")
    file.writelines("#SBATCH --ntasks-per-node="+str(num_processors)+"\n")
    file.writelines("mpirun -np "+str(num_processors)+" "+fepx_path+"\n")
    file.writelines("exit 0")
    file.close()
    #os.system(f"sbatch --job-name={name} --hint=nomultithread run.sh")

##
def quat_prod(q1, q2,debug=False):
       # Quaternion Product
       # input a q1 and 4*1
       a=q1[0]
       b=q2[0]
       avect=np.array(q1[1:])
       bvect=np.array(q2[1:])
       #
       dotted_val=np.dot(avect,bvect)
       crossed_val=np.cross(avect, bvect)
       #
       ind1 =  a* b - dotted_val
       v    =  a*bvect + b*avect +crossed_val
       #
       if debug:
              print(v)
       quat = np.array([ind1,v[0],v[1],v[2]])
       #
       if quat[0]<0:
              quat=-1*quat
       #print(quat)
       return quat
#
def sampled_trajectories(cur_path,offset=0,sampled_elts="",sample_dense=1,debug=True,sim=1,end_path=""):
    #
    #
    aniso = ["125", "150", "175", "200", "300", "400"]
    slips = ["2", "4", "6"]
    step = 27
    #
    if sim!="isotropic":
        sim_num = int(sim)-1
        slip = slips[int((sim_num)/30)]
        set_num = str(int((sim_num%30)/6)+1)
        ani = aniso[int(sim_num%6)]
        name = "Cube_"+ani+"_"+slip+"ss_set_"+set_num
        print(name,sim_num)
        #exit(0)
    else:
        name = "Cube_control"
    #
    print(name) 
    print("--opened path ", cur_path)
    if debug:
        return "done"
    sample_num = 100
    start = 500
    if sampled_elts=="":
        sampled_elts= [i for i in range(offset+start,start+sample_num)]
        sampled_elts = sampled_elts[::sample_dense]
    else:
        sampled_elts=sampled_elts
    #
    #reports = open(path+'/post.report','r')
    num_steps = 27#int(reports.readlines()[8][15:])
    print("--opened path ", cur_path)
    print(num_steps,"======")
    #
    cur_path +="/results/elsets/ori/ori.step"
    init= open(end_path+'/ini','w')
    step_0 = open(cur_path+'0','r').readlines()
    for i in sampled_elts:
        init.write(step_0[i])

    print("\n---wrote to ",end_path+'/ini')
    final= open(end_path+'/fin', 'w')
    step_n = open(cur_path+str(num_steps),'r').readlines()
    for i in sampled_elts:
        final.write(step_n[i])

    print(os.getcwd())
    print("write to file ",end_path+'/'+name)
    all= open(end_path+'/'+name,"w")
    all.close()
    all= open(end_path+'/'+name,'a+')
    for j in range(num_steps+1):
        print(cur_path+str(j))
        curr= open(cur_path+str(j),'r').readlines()
        for i in sampled_elts:
            all.write(curr[i])
            #print(curr[i])
    
    print(sampled_elts)
    #exit(0)
    all.close()