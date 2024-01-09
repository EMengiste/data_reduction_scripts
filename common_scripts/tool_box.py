#------------------------------------------------------------------------
from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
import os
import time
import math
from math import cos,sin,acos,degrees,pi


def to_matrix(arr):
    # if arr.shape ==(6,):
    row1 = [arr[0],arr[-1],arr[-2]]
    row2 = [arr[-1],arr[1],arr[-3]]

    row3 = [arr[-2],arr[-3],arr[2]]
    matrix = [row1 ,row2, row3]
    return matrix
    # else:
    #     matrix = []
    #     for array in arr:
    #         row1 = [array[0],array[-1],array[-2]]
    #         row2 = [array[-1],array[1],array[-3]]
    #         row3 = [array[-2],array[-3],array[2]]
    #         matrix.append(np.array([row1 ,row2, row3]))
    #     return matrix
        
def von_mises_stress(stress):
    s11,s22,s33,s23,s13,s12 = stress
    ans =  (s11-s22)**2
    ans += (s22-s33)**2
    ans += (s33-s11)**2
    ans += 6*(s12**2 +s23**2 + s13**2)
    ans /=2
    return ans**0.5
def deviatoric(value):
    ## assume voight notation
    hydro= sum(value[:3])/3
    value[0] = value[0]-hydro
    value[1] = value[1]-hydro
    value[2] = value[2]-hydro
    return value

def find_equivalent(value):
    s11,s22,s33,s23,s13,s12 = deviatoric(value)
    ans =  (s11-s22)**2
    ans += (s22-s33)**2
    ans += (s33-s11)**2
    ans += 6*(s12**2 +s23**2 + s13**2)
    ans /=2
    return ans**0.5

def stress_triaxiality(stress):
    #
    stress_mat= to_matrix(np.array(stress))
    #
    #
    vm_stress = von_mises_stress(np.array(stress))
    eig_val= np.linalg.eigvals(stress_mat)
    #
    #
    psi =(eig_val[0]+eig_val[1]+eig_val[2])/(3*vm_stress)
    #
    #
    return psi
#
def find_nearest(a, a0):
    # https://stackoverflow.com/a/2566508
    "Element in nd array `a` closest to the scalar value `a0`"
    a=np.array(a)
    idx = np.abs(a - a0).argmin()
    return idx
### yield calculation for simulation data
def find_yield(stress, strain, offset="",number=""):
    load_steps= len(strain)
    #
    if offset == "":
        offset = 0.002
    if number == '':
        E = stress[1]/ strain[1]
        index = ''
        print("e=",E)
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

def quat_of_angle_ax(angle, raxis):
       # angle axis to quaternion
       #
       half_angle = 0.5*angle
       #
       cos_phi_by2 = cos(half_angle)
       sin_phi_by2 = sin(half_angle)
       #
       rescale = sin_phi_by2 / np.sqrt(np.dot(raxis,raxis))
       quat = np.append([cos_phi_by2],np.tile(rescale,[3])*raxis)
       #
       if cos_phi_by2<0:
              quat = -1*quat
       #
       #
       return quat
#
def generate_tesr(n,name,source_dir="",scale=1000,source_code="neper",from_nf=False,from_ff=True,fin_dir="",options={"mode" :"run"}):
    print("\n===")
    tesselation= name
    if from_ff:                
        file1= open(fin_dir+n+'centroids', 'w')
        file2= open(fin_dir+n+'centvols', 'w')
        file3= open(fin_dir+n+'radii', 'w')
        file4 = pd.read_csv(source_dir+name)


        file4.sort_values(by=['Z','X','Y'], ascending=[True, True, True], inplace= True)

        x= np.sort(pd.unique(file4['X']))
        y= np.sort(pd.unique(file4['Y']))
        z= np.sort(pd.unique(file4['Z']))

        file4['X']-= x[0]
        file4['Y']-= y[0]
        file4['Z']-= z[0]

        x= file4['X']/scale
        z= file4['Y']/scale
        y= file4['Z']/scale
        try:
            weight= file4['Grain Radius']/scale
        except:             
            weight= file4['Radius']/scale

        for i in range(len(file4)):
            file1.write(str(x[i])+' '+str(y[i])+' '+str(z[i])+'\n')
            file2.write(str(x[i])+' '+str(y[i])+' '+str(z[i])+' '+str((4*(weight[i]**3))*(math.pi/3))+'\n')
            file3.write(str(weight[i])+'\n')
        file1.close()
        file2.close()
        domain=str(max(x))+","+str(max(y))+","+str(max(z))
        out =[len(weight), domain]
        print(out)
        return out    
    if from_nf:                
        file3= open(fin_dir+tesselation+'.tesr', 'w')
        file4 = pd.read_csv(source_dir+name)


        file4.sort_values(by=['Z','X','Y'], ascending=[True, True, True], inplace= True)

        x= np.sort(pd.unique(file4['X']))
        y= np.sort(pd.unique(file4['Y']))
        z= np.sort(pd.unique(file4['Z']))

        file4['X']-= x[0]
        file4['Y']-= y[0]
        file4['Z']-= z[0]

        x= file4['X']/scale
        z= file4['Y']/scale
        y= file4['Z']/scale



        for i in range(len(file4)):
            file1.write(str(x[i])+' '+str(y[i])+' '+str(z[i])+'\n')
            file2.write(str(x[i])+' '+str(y[i])+' '+str(z[i])+' '+str((4*(weight[i]**3))*(math.pi/3))+'\n')
            file3.write(str(weight[i])+'\n')
        file1.close()
        file2.close()
        domain=str(max(x))+","+str(max(y))+","+str(max(z))
        out =[len(weight), domain]
        print(out)
        return out

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
def visualize(input_source,source_code="neper",outname="default_img",options={"mode" :"run"}):
    commands = []
    input_name= input_source.split("/")[-1]
    neper_command=source_code+" -V "+input_source+" "
    # populate the commands using the optional input dictionary
    for i in options:
        if i[0]== "-":
            neper_command+=" "+i+' '+options[i]
            commands.append(i+' '+options[i])
    neper_command+=" -print "+outname
    if options["mode"]=="debug":
        #print(input_source)
        print(os.getcwd())
        pprint(commands)
        print("Visualization command:\n",neper_command)
    elif options["mode"]=="run":
        print("Visualization command:\n",neper_command)
        if os.path.exists(outname+".png"):
            print("Image already exists",outname+".png")
        else:
            print("Image doesn't exist generating new")
            print(os.getcwd())
            os.system(neper_command+" ")#> vis_output"+input_name)
    elif options["mode"]=="rerun":
        os.system(neper_command+" > vis_output")
#
#
if __name__=="__main__":
    print("trying")
    print(os.getcwd())
    # exit(0)
    source_dir="/home/etmengiste/jobs/aps/full_aps_data_set/Fe9Cr HEDM data repository/Fe9Cr-61116 (unirr)/In-situ ff-HEDM/"
    fin_dir="/home/etmengiste/jobs/aps/far_field_data_presentation/"
    
    opt= generate_tesr("1",name="In-situ FF Parent Grain Data Pre-def.csv",source_dir=source_dir,fin_dir=fin_dir)
    print(opt)
    generate_tess("1","border",fin_dir,source_code="neper",options={"-domain":" 'cube("+opt[1]+")'","mode" :"run"})
    os.chdir(fin_dir)
    visualize("border.tess,1centroids",source_code="neper",options={"mode" :"run",
                                                                    "-datacelltrs": "1.23",
                                                                    "-datapointrad": "1radii"})
    tess = generate_tess(opt[0],"FF_predef",fin_dir,source_code="neper",options={"-domain":" 'cube("+opt[1]+")'",
                                                                       "-morpho":"'centroidsize:file(1centvols)'",
                                                                       "mode" :"run"})
    visualize(tess,source_code="neper",options={"mode" :"run",
                                                "-datacellcol":"ori"})
    exit(0)
    print("trying")
    print(os.getcwd())
    source_dir="/home/etmengiste/jobs/aps/full_aps_data_set/Fe9Cr HEDM data repository/Fe9Cr-61116 (unirr)/ex-situ nf-HEDM/"
    fin_dir="/home/etmengiste/jobs/aps/far_field_data_presentation/"

    opt= generate_tesr("Near_field",name="Ex-situ NF Deformed Grain Data.csv",source_dir=source_dir,fin_dir=fin_dir)
    ####
    exit(0)

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

def norm(vect):
    value= 0
    for i in vect:
        value+=(i**2)
    mag=(value)**0.5
    return mag
def normalize_vector(vect,magnitude=False):
    value= 0
    final = vect
    for i in vect:
        value+=(i**2)
    mag=(value)**0.5
    for i in range(len(vect)):
        final[i] = final[i]/mag
    if magnitude:
        return [final,mag]
    else:
        return final
  #
 #
#
def Cubic_sym_quats():
    # Generate Cubic symetry angle axis pairs for the cubic fundamental region
    pi = math.pi
    AngleAxis =  np.array([[0.0     , 1 ,   1,    1 ],   # % identity
                    [pi*0.5  , 1 ,   0,    0 ],   # % fourfold about x1
                    [pi      , 1 ,   0,    0 ],   #
                    [pi*1.5  , 1 ,   0,    0 ],   #
                    [pi*0.5  , 0 ,   1,    0 ],   # % fourfold about x2
                    [pi      , 0 ,   1,    0 ],   #
                    [pi*1.5  , 0 ,   1,    0 ],   #
                    [pi*0.5  , 0 ,   0,    1 ],   # % fourfold about x3
                    [pi      , 0 ,   0,    1 ],   #
                    [pi*1.5  , 0 ,   0,    1 ],   #
                    [pi*2/3  , 1 ,   1,    1 ],   # % threefold about 111
                    [pi*4/3  , 1 ,   1,    1 ],   #
                    [pi*2/3  ,-1 ,   1,    1 ],   # % threefold about 111
                    [pi*4/3  ,-1 ,   1,    1 ],   #
                    [pi*2/3  , 1 ,  -1,    1 ],   # % threefold about 111
                    [pi*4/3  , 1 ,  -1,    1 ],   #
                    [pi*2/3  ,-1 ,  -1,    1 ],   # % threefold about 111
                    [pi*4/3  ,-1 ,  -1,    1 ],   #
                    [pi      , 1 ,   1,    0 ],   # % twofold about 110
                    [pi      ,-1 ,   1,    0 ],   #
                    [pi      , 1 ,   0,    1 ],   #
                    [pi      , 1 ,   0,   -1 ],   #
                    [pi      , 0 ,   1,    1 ],   #
                    [pi      , 0 ,   1,   -1 ]])

    cubic_sym = np.array([quat_of_angle_ax(a[0],a[1:]) for a in AngleAxis])
    return cubic_sym

def rod_to_angle_axis(rod):
       # Rodrigues vector to Quaternion
       #
       norm,mag = normalize_vector(rod,magnitude=True)
       omega= 2*math.atan(mag)
       return [norm,omega]

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

def normalize_vector(vect,magnitude=False):
    value= 0
    final = vect
    for i in vect:
        value+=(i**2)
    mag=(value)**0.5
    for i in range(len(vect)):
        final[i] = final[i]/mag
    if magnitude:
        return [final,mag]
    else:
        return final
  #
 #
#
###
def ret_to_funda(quat="",rod="", sym_operators=Cubic_sym_quats(),debug=False):
       #    Return quaternion to the fundamental region given symerty 
       #        operatiors
       #
       if str(rod)!="":
            quat = rod_to_quat(rod)
       #
       m = len(sym_operators)
       n = 1
       # if passing a set of symetry operators make sure to [quat]
       tiled_quat = np.tile(quat,( m,1))
       #
       reshaped_quat=tiled_quat.reshape(m*n,4,order='F').copy()
       #sym_operators=sym_operators.T
       equiv_quats = quat_prod_multi(reshaped_quat,np.tile(sym_operators,(1,n)))
       return equiv_quats
#
def dif_degs_local(start,fin,debug=False):
	# Get basis mats
	q_ij,q_ji = rot_mat(start,fin)
	#print("misorientation matrix",q_ij)
	# Get misorination mat
	v1 = normalize_vector(R.from_matrix(q_ij).as_quat())
	v1=[v1[3],v1[0],v1[1],v1[2]]
	#v1 = normalize_vector(rot_to_quat_function(q_ij,option=""))
	#print("quaternion of misori",v1)
	# Get fuda
	r1 = ret_to_funda(v1)
	#print("funda quaternion of misori",r1)
	# Get degs
	thet_ij =2* math.degrees(math.acos(min([v1[0],1])))

	return thet_ij
###
def rod_to_quat(val,debug=False):
       # Rodrigues vector to Quaternion
       #
       norm,mag = normalize_vector(val,magnitude=True)
       omega= 2*math.atan(mag)
       if debug:
              print("an ax",quat_of_angle_ax(omega,norm))
              print(omega)
       s= math.sin(omega/2)
       c= math.cos(omega/2)
       values = np.array([c, s*norm[0],s*norm[1], s*norm[2]])
       return values
##
def rot_mat(arr1,arr2):
    #   Find the roation matrix from a basis matrix 
    #       Q_ij = arr1 => arr2
    #       Q_ji = arr1 => arr2
    R_ij = []
    R_ji = []
    if len(arr1) ==len(arr2):
        for a in arr1:
            temp = []
            for b in arr2:
                    temp.append(np.dot(a,b))
            R_ij.append(temp)
        #
        for b in arr1:
            temp = []
            for a in arr2:
                    temp.append(np.dot(a,b))
            R_ji.append(temp)                     
        return [np.array(R_ij),np.array(R_ji)] 
    else:  
            print("not same size")
##
def quat_misori(q1, q2):     
       a=-q1[0]
       b=q2[0]
       avect=np.array(q1[1:])
       bvect=np.array(q2[1:])
       #
       dotted_val=np.dot(avect,bvect)
       ind1 =  a* b - dotted_val
       return degrees(acos(ind1))*2
       
def quat_prod(q1, q2):
       # Quaternion Product
       # input a q1,q2 4*1
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
       quat = np.array([ind1,v[0],v[1],v[2]])
       #
       if quat[0]<0:
              quat=-1*quat
       #print(quat)
       return quat
       #

def quat_to_angle_axis(q1):
    omega = 2*acos(q1[0])
    s= sin(omega/2)
    norm = [ i/s for i in q1[1:]]
    return [omega,norm]

def quat_prod_multi(q1,q2):
	quat =[]
	for i in range(len(q1)):
		quat.append(quat_prod(q1[i],q2[i]))
	max = 0
	max_ind=0
	for ind,q in enumerate(quat):
		if q[0]<0:
			quat[ind] = -1*quat[ind]
		if ind==0:
			max= quat[ind][0]
			max_ind=ind
		elif quat[ind][0]>max:
			#print("larger")
			#print(quat[ind])
			max=quat[ind][0]
			max_ind=ind
		#
	value = normalize_vector(quat[max_ind])
	#print("--------",value)
	return value
##   
def quat_of_angle_ax(angle, raxis):
       # angle axis to quaternion
       #
       half_angle = 0.5*angle
       #
       cos_phi_by2 = math.cos(half_angle)
       sin_phi_by2 = math.sin(half_angle)
       #
       rescale = sin_phi_by2 / np.sqrt(np.dot(raxis,raxis))
       quat = np.append([cos_phi_by2],np.tile(rescale,[3])*raxis)
       if cos_phi_by2<0:
              quat = -1*quat
       #
       #
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