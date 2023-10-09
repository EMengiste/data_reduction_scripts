import os
#from ezmethods import *
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R
import numpy as np
import math
import time
import multiprocessing 
import pandas as pd
# Latex interpretation for plots
SIZE=35
plt.rc('font', size=SIZE)            # controls default text sizes
plt.rc('axes', titlesize=SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)       # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)      # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)      # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)      # legend fontsize
plt.rc('figure', titlesize=SIZE)     #
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.figsize'] = 20,20

##
def pprint(arr, preamble="\n-+++",max=50):
    space= "                                                   "
    for i in arr:
        i=str(i)
        if isinstance(arr,dict):
            val = preamble+i+": "+str(arr[i])+space
            #                 :boundary_conditions: uniaxial_grip                 '
            print(val[:max]+"|||||------|")
        else:
            val = i+ space
            print(preamble+val[:max]+"|||||------|")
        preamble="    "
    #
#
#
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
#
def to_matrix(arr):
    if arr.shape ==(6,):
        row1 = [arr[0],arr[-1],arr[-2]]
        row2 = [arr[-1],arr[1],arr[-3]]
        row3 = [arr[-2],arr[-3],arr[2]]
        matrix = [row1 ,row2, row3]
        return matrix
    else:
        matrix = []
        for array in arr:
            row1 = [array[0],array[-1],array[-2]]
            row2 = [array[-1],array[1],array[-3]]
            row3 = [array[-2],array[-3],array[2]]
            matrix.append(np.array([row1 ,row2, row3]))
        return matrix
        
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
        for b in arr1:
            temp = []
            for a in arr2:
                    temp.append(np.dot(a,b))
            R_ji.append(temp)                     
        return [np.array(R_ij),np.array(R_ji)] 
    else:  
            print("not same size")
#
##
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
#
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
###
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

def quat_prod_funda(q1,q2):
	quat =[]
	for i in range(len(q1)):
		quat.append(quat_prod(q1.T[i],q2.T[i]))
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
##
def ret_to_funda(quat="", sym_operators=Cubic_sym_quats(),debug=False):
       #    Return quaternion to the fundamental region given symerty 
       #        operatiors
       #
       #
       m = len(sym_operators)
       n = 1
       # if passing a set of symetry operators make sure to [quat]
       tiled_quat = np.tile(quat,(1,m))
       #
       reshaped_quat=tiled_quat.reshape(4,m*n,order='F').copy()
       sym_operators=sym_operators.T
       if debug:
              print("\n\n\n+++++ ret to funda ++++")
              pprint(tiled_quat,preamble="tiled")
              print(tiled_quat.shape)
              pprint(reshaped_quat,preamble="reshaped")
              print("old shape",sym_operators.shape)
              pprint(sym_operators,max = 200)
              print(sym_operators.shape)
       equiv_quats = quat_prod_funda(reshaped_quat,np.tile(sym_operators,(1,n)))
       #pprint(equiv_quats)
       #exit(0)
       return equiv_quats
#
###### all above are dependencies with the difference in degrees between two rotation matricies
###### calculated using the function below to calculate the values
#
def dif_degs_local(start,fin,debug=False):
    # Get basis mats
    q_ij,q_ji = rot_mat(start,fin)
    # Get misorination mat
    v1 = normalize_vector(R.from_matrix(q_ij).as_quat())
    v1= [v1[3],v1[0],v1[1],v1[2]]
    #v1 = normalize_vector(rot_to_quat_function(q_ij,option=""))
    # Get fuda
    r1 = ret_to_funda(v1)
    # Get degs
    thet_ij =math.degrees(math.acos(min([r1[0],1])))
    if debug:
        print(np.dot(q_ij.T,start[0]))
        print("---")
        print(np.dot(q_ij.T,fin[0]))
        r2 = ret_to_funda(R.from_matrix(q_ji).as_quat())
        thet_ji =math.degrees(math.acos(r2[0]))
        return [thet_ij,thet_ji]
    
    return thet_ij

def calc_grain_stress_misori(params):
    #print(params)
    sim_ind,base,home,basic,step =params
    sim_ind = int(sim_ind)
    base= int(base)
    print(params)
    print(simulations)
    #exit(0)
    aniso = ["125", "150", "175", "200", "300", "400"]
    sets= ["1", "2", "3", "4", "5"]
    slips = ["2","4","6"]
    sim_name = simulations[sim_ind]
    sim = fepx_sim("Cube.sim",path=home+sim_name+"/Cube.sim")
    num_sets = len(sets)
    set_name = str(sets[int(sim_ind/base)%num_sets])
    slip = str(slips[int(sim_ind/(base*num_sets))])
    dom = "CUB"
    grain_start =1
    grain_end = 2000
    temp = []
    for id in range(grain_start,grain_end):
            stress_iso = sim_iso.get_output("stress",step=step,res="elsets",ids=[id])
            stress_mat= to_matrix(np.array(stress_iso))
            eig_val_iso, eig_vect_iso = np.linalg.eig(stress_mat)
            sorting = np.argsort(eig_val_iso)
            eig_val_iso = eig_val_iso[sorting]
            eig_vect_iso = eig_vect_iso[sorting]
            #print("eig_vects iso",eig_vect_iso)
            stress = sim.get_output("stress",step=step,res="elsets",ids=[id])
            stress_mat= to_matrix(np.array(stress))
            eig_val, eig_vect = np.linalg.eig(stress_mat)
            sorting = np.argsort(eig_val)
            eig_val = eig_val[sorting]
            eig_vect = eig_vect[sorting]
            #
            temp.append(dif_degs_local(eig_vect_iso,eig_vect))
            #pprint(eig_vect_iso,preamble="iso")
            #pprint(eig_vect,preamble="ani")
            #print(temp)
    #print(aniso[ind])
    if basic=="True":
       name = "DOM_"+dom+"_"+sim_name
    else:
        name = "DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set_name+"_ANISO_"+aniso[sim_ind%6]
    print(name)
    vals = [name ,np.mean(temp),np.std(temp)]
    del sim
    return vals

def plot_mean_data(NAME,ylims=[4.9,15.1],y_label="",
                       unit="",y_ticks ="",y_tick_lables="",debug=False):
    norm =False
    result ={}
    results = pd.read_csv(NAME+".csv").transpose()
    for i in results:
        result[results[i]["case"]]= [float(results[i]["mean"]),float(results[i]["std"])]
        #print(results[i]["case"],result[results[i]["case"]])
    DOMAIN = [ "CUB"]
    DOM = [ "cubic"]
    NSLIP  = ["2","4", "6"]
    #NSLIP =["2"]
    ANISO  = ["125", "150", "175", "200", "400"]
    SETS    = ["1", "2", "3", "4", "5"]
    an = ["1.25", "1.50", "1.75", "2.00", "3.00", "4.00"]
    #SETS = ["1"]
    sets    =  ["solid","dotted","dashdot",(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))]
    #
    ###
    aniso = [125, 150, 175, 200, 300, 400]
    differences= {}
    difs2={}

    if debug:
        NSLIP =["2"]
        DOMAIN = ["CUB"]
        DOM = ["cubic"]

    for dom in DOMAIN:
        fig, axs = plt.subplots(1, 3,sharex="col",sharey="row",figsize=( 23,8))
        for slip in NSLIP:
            #
            ax0= axs[NSLIP.index(slip)]
            #ax1= axs[1][NSLIP.index(slip)]
            #
            ax0.set_title(slip+" slip systems strengthened")
            #
            ax0.set_xlim([115,410])
            #ax1.set_xlim([90,410])
            if ylims!="":
                ax0.set_ylim(ylims)
                #ax1.set_ylim([0.95,2.0])
            else:
                ax0.set_ylim([4.6,15.1])
                #ax1.set_ylim([-0.4,10.4])
            #fig, ax = plt.subplots(1, 1,figsize=(10,12))
            for set,line in zip(SETS,sets):
                #
                index1 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_125"]
                index2 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_150"]
                index3 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_175"]
                index4 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_200"]
                index5 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_300"]
                index6 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_400"]

                list1 = [index1[0], index2[0],
                    index3[0], index4[0], index5[0],index6[0]]
                list2 =[index1[1], index2[1],
                    index3[1], index4[1], index5[1],index6[1]]
                
                differences[dom+"_"+slip+"_"+set] = list1
                difs2[dom+"_"+slip+"_"+set] = list2
                #print(list)
                #
                if norm == True:
                    list1 = [i/result["DOM_"+dom+"_ISO"][0] for i in list1]
                    list2 = [i/result["DOM_"+dom+"_ISO"][1] for i in list2]
                #### Start of plotting code
                #
                ax0.plot(aniso,list1,"k",lw=2,linestyle=line,
                    label="Set "+str(set))
                #ax1.plot(aniso,list2,"k",lw=2,linestyle=line,
                #    label="Set "+str(set))
                #
                marker_size =130
                #
                for a, l, l2 in zip(aniso, list1, list2):
                    ax0.scatter(a, l, marker="o", s=marker_size,edgecolor="k",color= "k")
                    #ax1.scatter(a, l2, marker="o", s=marker_size,edgecolor="k",color= "k")
                #
                ax0.set_xticks(aniso)
                ax0.set_xticklabels(an,rotation=90)

                #ax1.set_xticks(aniso)
                #ax1.set_xticklabels(an,rotation=90)
        if norm and unit!="":
            unit = "(-)"
        axs[0].set_ylabel(y_label,labelpad=25)

        if y_tick_lables !="":
            axs[0].set_yticks(y_ticks)
            axs[0].set_yticklabels(y_tick_lables)
            axs[0].set_ylabel(y_label+unit,labelpad=25)
        ###
        #axs[0].legend()
        #
        # Put title here
        #title = "Deviation from Taylor Hypothesis: \n"+str(DOM[DOMAIN.index(dom)])+" domain, "+str(slip)+" slip systems\n"
        #
        deg = "$^{\circ}$"
        #
        x_label = f'$p$ (-)'
        #y_label = f'Normalized Misorienation ({deg})'
        fig.supxlabel(x_label,fontsize=SIZE)
        fig.subplots_adjust(left=0.09, right=0.98,top=0.9, bottom=0.2, wspace=0.07, hspace=0.1)
        fig.savefig(NAME+"_"+str(DOM[DOMAIN.index(dom)])+"_mean.png",dpi=400)
        print("generated "+NAME+"_"+str(DOM[DOMAIN.index(dom)])+"_mean.png")


######## below this would require the fepx_sim object and access to schmid

class fepx_sim:
    #
    #
    # name: name of the simulation
    # path: path to the simulation raw output
    # debug_config_input: config input debugging
    #
    #
    #####-------
    #
    def __init__(self,name,path="",debug_config_input =False):
        self.name=name
        print("-----------------------------------##")
        print("                   |||||------##")
        print("                   |||||------object < "+self.name+"> creation")
        print("                   |||||------##")
        print("---------------------------------##")
        if path =="":
            path = os.getcwd()
        self.path = path
        #
        self.name=name
        #
        #   Simulation attributes
        #
        #--# Optional Input
        #
        self.optional_input = {}
        #
        #--# Material Parameters
        #
        self.material_parameters = {}
        #
        #--# Deformation History
        #
        self.deformation_history = {}
        #
        #--# Boundary Condition
        #
        self.boundary_conditions = {}
        #
        #--# Printing Results
        #
        self.print_results  = []
        #
        #
        ##@ Status checks
        self.sim=""
        self.is_sim=False
        self.results_dir=os.listdir(path)
        self.completed = "post.report" in self.results_dir
        self.has_config = "simulation.config" in self.results_dir
        if path[-4:] == ".sim":
            self.is_sim=True
            self.post_processed=True
            pass
        else:
            self.post_processed=os.path.isdir(path+".sim")
        #
        #
        #
        #
        # If the config file exists poulate the attributes
        #
        if self.is_sim:
            print("########---- opening "+path+"/simulation.config")
            self.post_process()
            config = open(path+"/inputs/simulation.config").readlines()

        elif self.has_config and not self.is_sim:
            #print(self.name,"has config")
            print("########---- opening "+path+"/simulation.config")
            config = open(path+"/simulation.config").readlines()
        else:
            #
            ##@ Status checks
            self.sim=""
            self.results_dir=os.listdir(path)
            self.completed = "post.report" in self.results_dir
            self.has_config = "simulation.config" in self.results_dir
            self.post_processed=os.path.isdir(path+".sim")
            #
            # If the config file exists poulate the attributes
            #
            if self.has_config:
                #print(self.name,"has config")
                print("########---- opening "+path+"/simulation.config")
                config = open(path+"/simulation.config").readlines()
            else:
                print(self.name,"has not_config initalization aborted")
                return
                #
            #
        #
        #
        for i in config:
            if i.startswith("##"):
                #print(i)
                current = i
            if "Optional Input" in current:
                if len(i.split())>1 and "Optional Input" not in i:
                    option= i.split()
                    self.optional_input[option[0]]=option[1]
                    #print(option)
                #
            #
            if "Material Parameters" in current:
                if len(i.split())>1 and not "Material Parameters" in i:
                    option= i.split()
                    self.material_parameters[option[0]]=option[1:]
                    #print(option)
                #
            #
            if "Deformation History" in current:
                if len(i.split())>1 and "Deformation History" not in i:
                    option= i.split()
                    self.deformation_history[option[0]]=option[1]
                    #print(option)
                #
            #
            if "Boundary Condition" in current:
                if len(i.split())>1 and "Boundary Condition" not in i :
                    option= i.split()
                    self.boundary_conditions[option[0]]=option[1]
                    #print(option)
                #
            #
            if "Printing Results" in current:
                if len(i.split())>1 and "Printing Results" not in i:
                    option= i.split()
                    self.print_results.append(option[1])
                    #print(option)
                #
            #
        if debug_config_input:
            pprint(self.optional_input)
            pprint(self.material_parameters)
            pprint(self.deformation_history)
            pprint(self.boundary_conditions)
            pprint(self.print_results)
        #
    #
    def get_num_steps(self):
        if self.deformation_history["def_control_by"]=="uniaxial_load_target":
            num =self.deformation_history["number_of_load_steps"]
            #print("number_of_load_steps",num)
        if self.deformation_history["def_control_by"]=="uniaxial_strain_target":
            num =self.deformation_history["number_of_strain_steps"]
            #print("number_of_strain_steps",num)
        if self.deformation_history["def_control_by"]=="triaxial_constant_strain_rate":
            num =self.deformation_history["number_of_strain_steps"]
            #print("number_of_strain_steps",num)
        if self.deformation_history["def_control_by"]=="triaxial_constrant_load_rate":
            num =self.deformation_history["number_of_strain_steps"]
            #print("number_of_strain_steps",num)
        return int(num)
        #
    #
    #
    def get_results(self,steps=[],res="mesh"):
        #
        # Available results
        print("____Results__availabe___are:")
        #
        pprint(self.print_results, preamble="\n#__|")
        #
        node_only = ["coo","disp","vel"]
        mesh= [i for i in self.print_results if i not in node_only ]
        #
        print("\n____Getting results at "+res+" scale\n   initializing results\n")
        #
        num_steps=self.get_num_steps()
        if res == "mesh":
            length= len(mesh)

        #
        #
        results_dict= {self.name: "sim","num": num_steps}
        #
        pprint(results_dict)
        self.json = self.path+"/"+self.name+".txt"
        #
        #
        if self.json in os.listdir(self.path):
            converter_file= open(self.json,"r")
            print("json file exists parsing")
            results_dict = json.load(converter_file)
            converter_file.close()
            return results_dict
        else:
            for index in range(length):
                result=mesh[index]
                #print("\n\n--===== start"+result+"\n")
                steps =[]
                fill = '█'
                percent = round(index / float(length-1),3)
                filledLength = int(40 * percent)
                percent*=100
                bar = fill * filledLength + '-' * (length - filledLength)
                prefix=" \n\n== Getting <"+res+">results for <"+result+">\n--step<"

                for step in range(num_steps):
                    prefix+=str(step)
                    if step==10:
                        prefix+="\n"
                    try:
                        vals = [float(i) for i in self.get_output(result,step=str(step),res=res)]
                        steps.append(vals)
                        print(f'\r{prefix} |{bar}| {percent}% ')
                    except FileNotFoundError:
                        #print("file not found Trying nodes")
                        prefix=self.name+" \n\n===== Getting <nodes>results for <"+result+">----\n-----<"
                        try:
                            vals = [float(i) for i in self.get_output(result,step=str(step),res="nodes")]
                            steps.append(vals)
                            print(f'\r{prefix} |{bar}| {percent}% ')
                        except FileNotFoundError:
                            error = " youre outa luck"
                            print(f'\r{prefix+error} |{bar}| {percent}% ')
                prefix+= ">--------|\n+++\n+++"
                #print("--===== end"+result+"\n")
                results_dict[result]=steps
            with open(self.json,"w") as converter_file:
                converter_file.write(json.dumps(results_dict))
                self.results_dir=self.path+"/"+self.name+".txt"
            return results_dict
        #
        #
    #
    #
    def get_output(self,output,id=0,step= "0", res="",ids=[0],num_steps=0,debug=False):
        step = str(step)
        value = {}
        if res=="":
            res="mesh"
        #
        ##
        if output in ["coo","disp","vel"] and res !="nodes":
            print("invalid output try again")
            return 
        #
        # #       
        if step=="malory_archer":
            if num_steps==0:
                num_steps = self.get_num_steps()
            print(num_steps)
            vals = []
            #
            # #     
            for step in range(num_steps):
                print(str(step))
                value = self.get_output(output,ids=ids,step=step,res=res)
                vals.append(value)
            return vals
        #
        # #     
        if self.path[-4:] == ".sim":
            step_file = self.path+"/results/"+res+"/"+output+"/"+output+".step"+step
        else:
            step_file = self.path+".sim/results/"+res+"/"+output+"/"+output+".step"+step
                #
        # #     
        file = open(step_file)
        values=file.readlines()
        num_components= len(values[0].split())
        if ids =="all":
            ids= [i for i in range(len(values))]
        #print(ids)
        for id in ids:
            #print(id,"--------")
            value[str(id)] = [float(i) for i in values[id].split()]
            #pprint(value,max=1000)
        if len(ids)==1:
            return value[str(ids[0])]
        else:
            return [value[i] for i in value]
        #
       #
      #
     #
    #
    #
    def post_process(self,options="",debug=False):
        #
        if self.post_processed:
            print("Already post processed")
            end=".sim/.sim"
            if self.is_sim:
                end= "/.sim"
            if self.sim == "":
                values = {}
                with open(self.path+end) as file:
                    sim_file = [i.strip() for  i in file.readlines()]
                    for line in sim_file:
                        if line.startswith("***"):
                            print(line,"\n")
                        elif line.startswith("**"):
                            values[line[2:]]= sim_file[sim_file.index(line)+1].strip()
                self.sim= values
            if debug:
                pprint(values)
                print(values["general"][8])
            return
        #
        elif options!="":
            print(options.split("-res"))
            print("\n\n")
            print(os.getcwd())
            print(options)
            os.chdir(self.path)
            os.system(options)
            print("\n\n")
            with open(self.path+".sim/.sim") as file:
                self.sim=file.readlines()
                return
        #
        #
        elif not self.completed:
            print("simulation not done come back after it is")
            return
        #
       #
      #
     #
    #
    #
    def get_summary(self):
        return "stuff"
        #
       #
      #
     #
    #
    #
    def __del__(self):
        print("-----------------------------------##")
        print("                   |||||------##")
        print("                   |||||------object < "+self.name+"> destruction")
        print("                   |||||------##")
        print("---------------------------------##")
        #
       #
      #
     #
    #
    #
    #
   #
  #
 #
#
### 
iso_home="/home/etmengiste/jobs/aps/slip_study/"
home="/media/schmid_2tb_1/etmengiste/files/slip_system_study/"

simulations = os.listdir(home)
simulations.sort()
simulations.remove("common_files")
sim_iso = fepx_sim("Cube.sim",path=home+"isotropic/Cube.sim")
step= "27"
num_steps = sim_iso.get_num_steps()
#destination = "/home/etmengiste/jobs/aps/misori_test/"
destination=" "
steps = [str(i) for i in range(num_steps)]
pool = multiprocessing.Pool(processes=90)

set_start = 0
set_end = 5
headers = ["case","mean","std"]
data = [headers]
###
aniso = ["125", "150", "175", "200", "300", "400"]
slips = ["2","4","6"]
sets = ["1","2","3","4","5"]
num_sets = len(sets)
base = 6
dom = "CUB"
basic = False
#basic = True

def generate_data(step,simulations,home):
    print("starting code")
    tic = time.perf_counter()
        #
    print(tic)
    #   main_code
    num__base=6
    if basic:        
        simulations = ["isotropic_2","isotropic_3"]
        home = iso_home
        base = len(simulations)
        set_of_sims = [m for m in range(0,base,1)]
    else:
        base = len(simulations[:90])
        set_of_sims = [m for m in range(0,base,1)]
    print("----")
    sims = np.array([set_of_sims,np.tile(num__base,(base))
                    ,np.tile(home,(base)),np.tile(basic,(base))
                    ,np.tile(step,(base))]).T
    #value = pool.map(calc_grain_stress_delta,sims)
    #value = calc_grain_stress_delta(sims[-1])
    #value = calc_grain_stress_misori(sims[-1])
    #print("sims 01",sims[-1],value)
    #exit(0)
    headers = ["case","mean","std"]
    data = [headers]
    ###
    if basic:
        value = calc_grain_stress_misori(sims[0])
        pprint(value)
        value = calc_grain_stress_misori(sims[1])
        pprint(value)
    else:
        value = pool.map(calc_grain_stress_misori,sims)
        data +=value   
    toc = time.perf_counter()

    name = "calculation_stress_misori_step_"+step
    df1 = pd.DataFrame(data)
    df1.columns=df1.iloc[0]
    df1[1:].to_csv(destination+name+".csv")
    print("===")
    print("===")
    print("===")
    print(f"Generated data in {toc - tic:0.4f} seconds")
    #exit(0)

for step in steps[-1:]:
    print("===")
    print("===")
    generate_data(step,simulations,home)
    print("===")
    print("===")
    print("===")
    print("starting plotting")
    tic = time.perf_counter()

    ### misori plot code
    name = "calculation_stress_misori_step_"+step
    y_lab= "$\\phi_m$"
    ylims= [0.95,2.0]
    y_ticks = [5.00,7.50,10.00,12.50,15.00]
    y_tick_lables = ["5.00","7.50","10.00","12.50","15.00"]
    print("opened",destination+name+".csv")
    #ax = plt.figure().add_subplot(projection='3d')
    plot_mean_data(destination+name,ylims=[5,25],y_label=y_lab,debug=False)


    toc = time.perf_counter()
    print(f"Generated plot in {toc - tic:0.4f} seconds")

del sim_iso

exit(0)
