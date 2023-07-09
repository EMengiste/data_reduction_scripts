import os
import numpy as np
import json
import matplotlib.pyplot as plt
import matplotlib.transforms as mtransforms
import math
import pandas as pd
from sklearn.linear_model import LinearRegression
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R
SIZE=35
# Latex interpretation for plots
plt.rcParams.update({'font.size': SIZE})
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.subplot.left"] = 0.045
plt.rcParams["figure.subplot.bottom"] = 0.08
plt.rcParams["figure.subplot.right"] = 0.995
plt.rcParams["figure.subplot.top"] = 0.891
plt.rcParams["figure.subplot.wspace"] = 0.21
plt.rcParams["figure.subplot.hspace"] = 0.44
plt.rcParams['figure.figsize'] = 23,8#
plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  #
import numpy as np
home ="/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/"
home ="/media/schmid_1tb_2/etmengiste/aps_add_slip/"
ist= "/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/isotropic"
isotropic_home ="/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/isotropic/"

values = { "1=crss": [],
           "2=stress": [],
           "3=slip": [],
           "4=strain": [],
           "5=sliprate": [],
           "6=defrate_pl": [],
           "7=defrate_pl_eq": [],
           "8=elt_vol": [],
           "9=ori": [],
           "10=spinrate": [],
           "11=defrate": [],
           "12=strain_eq": [],
           "13=strain_el": [],
           "14=strain_el_eq": [],
           "15=strain_pl": [],
           "16=strain_pl_eq": [],
           "17=defrate_eq": [],
           "18=stress_eq": [],
           "19=velgrad": [],
           "20=work": [],
           "21=work_pl": [],
           "22=workrate": [],
           "23=workrate_pl": []}
  #
 #
#
##
CUB_110 = [[0, 1,-1],
           [1, 0 ,-1],
           [1,-1,0],
           [0, 1 , 1],
           [1, 0, 1],
           [1,-1,0],
           [0, 1, 1],
           [1, 0 ,-1],
           [1, 1, 0],
           [0, 1 ,-1],
           [1, 0, 1],
           [1, 1 ,0]]
  #
 #
#
##
CUB_111 = [[1, 1, 1],
           [1, 1, 1],
           [1, 1, 1],
           [1, 1,-1],
           [1, 1,-1],
           [1, 1,-1],
           [1,-1, 1],
           [1,-1, 1],
           [1,-1, 1],
           [1,-1,-1],
           [1,-1,-1],
           [1,-1,-1]]
  #
 #
#
sets    =  ["solid","dotted","dashdot",(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))]
an = ["Iso.", "1.25", "1.50", "1.75", "2.00", "3.00", "4.00"]
ani_marker= ["k",
             (10/255,10/255,10/255),
             (70/255,70/255,70/255),
             (150/255,150/255,150/255),
             (200/255,200/255,200/255),
             (215/255,215/255,215/255),
             (255/255,255/255,255/255)]
aps_home ="/home/etmengiste/jobs/aps/"
slips=["2","4", "6"]
x_label = f'$p$ (-)'
##
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
    def get_output(self,output,id=0,step= "0", res="",ids=[0]):
        step = str(step)
        value = {}
        if res=="":
            res="mesh"
        if output in ["coo","disp","vel"] and res !="nodes":
            print("invalid output try again")
            return        
        if step=="malory_archer":
            num_steps = self.get_num_steps()
            for step in range(num_steps):
                print(str(step))
        if self.path[-4:] == ".sim":
            step_file = self.path+"/results/"+res+"/"+output+"/"+output+".step"+step
        else:
            step_file = self.path+".sim/results/"+res+"/"+output+"/"+output+".step"+step
        
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
def diad(a,b):
    mat = []
    for i in range(3):
        row = []
        for j in range(3):
            row.append(a[i]*b[j])
        mat.append(row)
    return(mat)
    #
   #
  #
 #
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
##
def inner_prod(a,b):
    tot = 0
    for i in range(3):
        for j in range(3):
            tot+= a[i][j]*b[i][j]
    return tot
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
def normalize_vector(vect,magnitude=False):
    value= 0
    final = vect
    for i in vect:
        value+=(i**2)
    mag=math.sqrt(value)
    for i in range(len(vect)):
        final[i] = final[i]/mag
    if magnitude:
        return [final,mag]
    else:
        return final
  #
 #
#
##
def calculate_schmid(a,b):
    a= normalize_vector(a)
    b= normalize_vector(b)
    T = diad(a,b)

    P = 0.5*(T+np.transpose(T))
    #vect = vectorize(matrix)
    return P
    #print(calculate_inner(vect,vect))
  #
 #
#
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
 #
#
##
def list_sims(list,func="",option=[""]):
    array=[]
    for i in list:
        if func=="":
            print(i.material_parameters[option[0]])
        if func=="stress":
            print(i.get_output("strain",step=option[0],options=option[2])[option[1]])
    return array
  #
 #
#
##
def plotter(ax,x,y=""):
    if y=="":
        y=np.arange(len(x))
        ax.bar(y,x)
    else:
        ax.plot(x,y)
    return [x,y]
  #
 #
#
##
def directory_from_list(arr,splitter):
    dict={}
    for i in arr:
        dict[i.split(splitter)[0]]=i.split(splitter)[1]
    return dict
  #
 #
#
##
def plotting_space(x,y=[],layers=3,axis="",resolution=0,ylabel="",debug=False):
    print("-------|---------|---plotting_space--|---------|---------|---------|---------|")
    #print(x,y)
    values = {}
    #pprint(x)
    print("\nlist of vals---\n")
    #print(x)
    num_components= len(x)
    if y==[]:
        #print(num_components)
        wide=1/(len(x)+2)
        for m in range(len(x)):
            i=x[m]
            y= np.arange(len(i))
            spacing=wide*float(m)
            if axis!="":
                axis.bar(y+spacing,i,width=wide,edgecolor="k")
            if debug:
                print("======\n\n")
                print(m)
                print(y+spacing)
                print(float(x.index(i)))
                print("======\n\n")
                for j in range(len(x[0])):
                    print(str(j)+"  -------|---------|---------|---------|---------|---------|---------|---------|")
                    for i in range(len(x)):
                        ret= str(i)+" |"
                        print(ret)
                        #print(i)
                        spacer="  |"
                        print(x[i][j])
                        while float(x[i][j])>resolution:
                            ret   +="-.--|"
                            spacer+="---+|"
                            x[i][j]=float(x[i][j])-1
                        for _ in range(layers):
                            print(ret)
            axis.set_ylabel(ylabel)
            #print("-------|---------|---------|---------|---------|---------|---------|---------|")
  #
 #
#
##
def normalize(array,scale=1,maximum=1,absolute=False,debug=False):
    if debug:
        print("-------|---------|---------|---------|---------|---------|---------|---------|")
        print("-------|---------|---------|---------|---------|---------|---------|---------|")
        print("-------|---------|---------|---------|---------|---------|---------|---------|")

        print(array)
        print("maximum val "+str(maximum)+"------")
        print("-------|---------|---------|---------|---------|---------|---------|---------|")
        print("-------|---------|---------|---------|---------|---------|---------|---------|")
    # [f(x) if condition else g(x) for x in sequence]
    if isinstance(array[0],str):
        array=[float(i)
            if "d" not in i
            else float(i.split("d")[0])
            for i in array ]
    if isinstance(array[0],list):
        return array
    if absolute:
        array= [abs(i) for i in array]
    if maximum=="":
        maximum = max(array)
    return [scale*i/maximum for i in array]
  #
 #
#
def find_nearest(a, a0):
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
        stress_o = [(E * i)+ strain[0] for i in strain]
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
        stress_o= [(E * strain) + model.intercept_ for strain in strain]
        print('E=', model.coef_)
        print("b=",model.intercept_)
        print('index=',index)
        print('near=',find_nearest(stress,number))
        diff= np.array(stress)-np.array(stress_off)
        ind= np.abs(diff).argmin()
        Ystrain = float(strain[ind])
        Ystress = float(stress[ind])
    values ={"y_stress": Ystress,
             "y_strain": Ystrain,
             "stress_offset": stress_off}
    return values
  #
 #
#
#
def avg(arr):
    return sum(arr)/len(arr)
  #
 #
#
##
def slip_vs_aniso(sim_start,domain,slip_systems,show=False,target_dir="/home/etmengiste/jobs/aps/eff_pl_str/",base=5,iso_home=home,path=home,debug=False,save_plot=False,df="", ids=[0],ratios=[ 1.25, 1.5, 1.75, 2.0, 4],step = "28",res ="mesh"):
    P = [ calculate_schmid(CUB_111[i],CUB_110[i]) for i in range(12)]
    sim_iso= fepx_sim("name",path=iso_home+"isotropic/"+domain)
    #sim_iso.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
    sim_iso.post_process()
    baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])
    val =sim_iso.sim["**general"].split()
    slip_iso = normalize(sim_iso.get_output("slip",step=step,res=res,ids=ids),absolute=True)
    baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])
    #stress_iso = [normalize(sim_iso.get_output("stress",step=step,res=res,ids=ids)[str(i)]) for i in ids]
    del sim_iso
    print(path)
    #for i in range(6):
        #ax_stress = fig.add_subplot(hig,wid,12+i)
        #ax_stress.hist(stress_iso[str(i)],bins=bin_size,edgecolor="k")
        #ax_stress.set_title("Stress "+str(i))
        #stress_axies.append(ax_stress)
    #avg_slip_iso = []
    fig = plt.figure(1,figsize=[70,35])
    wid=6
    hig =4
    fig.clear()
    axies = []
    mean_iso = {}
    ax1= fig.add_subplot(hig,wid,13)
    ax2= fig.add_subplot(hig,wid,15)
    ax11= fig.add_subplot(hig,wid,14)
    ax22= fig.add_subplot(hig,wid,16)
    ax3= fig.add_subplot(hig,wid,17)
    ax4= fig.add_subplot(hig,wid,18)
    # vol avg
    ax5= fig.add_subplot(hig,wid,23)

    ax1.set_ylim(0,0.05)
    ax1.set_xlim(0.9,4.1)
    ax2.set_ylim(0,0.3)
    ax2.set_xlim(0.9,4.1)

    ax1.set_xlabel("ratio")
    ax2.set_xlabel("ratio")
    ax3.set_xlabel("ratio")
    ax5.set_xlabel("ratio")
    ax1.set_ylabel("$\\bar\\varepsilon^{p,a}$ ")
    ax11.set_ylabel("$\\bar\\varepsilon^{p,a}$ ")
    ax2.set_ylabel("$\\bar\\varepsilon^{p,u}$")
    ax22.set_ylabel("$\\bar\\varepsilon^{p,u}$")
    ax3.set_ylabel("normalized $\\bar\\varepsilon^{p}$")
    ax5.set_ylabel("normalized $\\bar\\varepsilon^{p}$")
    for index,item in enumerate(slip_systems):
        ax= fig.add_subplot(hig,wid,index+1)
        #
        ax.set_ylim(-0.0001,0.01)
        ax.set_xlim(0.9,4.1)
        #
        #print(slip_iso)
        #print(slip_iso[index])
        #
        ax.plot(1,slip_iso[index],"*k",ms=25)
        axies.append(ax)
        ax.set_xlabel("ratio")
        ax.set_ylabel("mesh $\gamma$")
        ax.set_title(item)
        #avg_slip_iso.append(np.average(slip_iso[str(index)])*10000000)
        #if index<6:
        #    ax2 = fig.add_subplot(hig,wid,18-index)
        #    ax2.plot(1,1)
        #    analysis_axes.append(ax2)
            #ax2.set_ylim[]
    print(path)
    simulations = os.listdir(path)
    simulations.sort()
    effective_pl_strain_unalt = []
    effective_pl_strain_alt = []

    vol_eff_pl_strain_unalt = []
    vol_eff_pl_strain_alt = []
    for index,sim in enumerate(simulations[sim_start:sim_start+base]):
        
        file=path+sim+"/"+domain[0:-4]+"_eff_pl_str.csv"
        if index <1:
            file_iso=path+sim+"/"+domain[:-4]+"iso_eff_pl_str.csv"
            print(file_iso)
            # Eff plast strain
            print("opening file ",file_iso)
            data = pd.read_csv(file_iso)
            tot_alt_iso = sum(data[" vol_eff_pl_alt"])
            tot_unalt_iso = sum(data[" vol_eff_pl_unalt"])
            print("--[+ total altered = ",tot_alt_iso)
            print("--[+ total unaltered = ",tot_unalt_iso)

            vol_eff_pl_strain_alt.append(tot_alt_iso/tot_alt_iso)
            vol_eff_pl_strain_unalt.append(tot_unalt_iso/tot_unalt_iso)
        sim= fepx_sim(sim,path=path+sim+"/"+domain)
        slip = normalize(sim.get_output("slip",step=step,res=res,ids=ids),absolute=True)
        #stress =  sim.get_output("stress",step=step,res=res,ids=ids)[1]
        mat_par = sim.material_parameters["g_0"]
        del sim
        strength = [ float(i.split("d")[0]) for i in mat_par]
        #exit(0)
        altered  =  []
        ratio=1
        for index,val in enumerate(strength):
            if val>baseline:
                altered.append(index)
                ratio = val/baseline
        total_altered=0
        total_unaltered=0
        total_altered_iso=0
        total_unaltered_iso=0
        #
        for index,item in enumerate(slip_systems):
            color = "blue"
            slip_val = slip[index]
            P_val = P[index]
            print("\n\n\nP")
            pprint(P_val)
            if index in altered:
                color="red"
                total_altered +=slip_val*P_val
                total_altered_iso +=slip_iso[index]*P_val
                print(" gamma on slip system", index)
                print(slip_iso[index])
                print("E_p,a ", index)
                print(total_altered)
            else:
                print(" gamma on slip system", index, slip_iso[index])
                print("E_p,u ", index)
                print(total_unaltered)
                total_unaltered+=slip_val*P_val
                total_unaltered_iso +=slip_iso[index]*P_val
            ax= axies[index]
            #print(mean)
            #ax.hist(normalize(slip[str(index)],absolute=True),bins=bins
            #        ,edgecolor="k",alpha= 0.2,color=color)
            ax.plot(ratio,slip_val,"o",ms=20,color=color)
        ax4.plot(ratio,sum(slip), "*k",ms=15)
        #
        if debug: # not going to print what you want check again
            print("\n\n\nsum of slip=",sum(slip))
            print("altered")
            print(total_altered)
            print("unaltered")
            print(total_unaltered)
            print("altered iso")
            print(total_altered_iso)
            print("unaltered iso" )
            print(total_unaltered_iso)
        #
        total_altered=math.sqrt((2/3)*inner_prod(total_altered,total_altered))
        total_unaltered=math.sqrt((2/3)*inner_prod(total_unaltered,total_unaltered))
        total_altered_iso=math.sqrt((2/3)*inner_prod(total_altered_iso,total_altered_iso))
        total_unaltered_iso=math.sqrt((2/3)*inner_prod(total_unaltered_iso,total_unaltered_iso))
        #
        if debug:
            print("\n\n\n\n++")
            print(total_unaltered)
            print(total_altered_iso)
            print(total_unaltered_iso)
        #
        effective_pl_strain_alt.append(total_altered/total_altered_iso)
        effective_pl_strain_unalt.append(total_unaltered/total_unaltered_iso)
        #
        # Eff plast strain
        print("opening file ",file)
        data = pd.read_csv(file)
        tot_alt = sum(data[" vol_eff_pl_alt"])
        tot_unalt = sum(data[" vol_eff_pl_unalt"])
        normalized_tot_alt =tot_alt/tot_alt_iso
        normalized_tot_unalt =tot_unalt/tot_unalt_iso
        print("--[+ total altered = ",normalized_tot_alt)
        print("--[+ total unaltered = ",normalized_tot_unalt)
        normalized_tot_alt =tot_alt/tot_alt_iso
        normalized_tot_unalt =tot_unalt/tot_unalt_iso
        vol_eff_pl_strain_alt.append(normalized_tot_alt)
        vol_eff_pl_strain_unalt.append(normalized_tot_unalt)
        #
        headers = [["tot_alt_msh","tot_unalt_msh","tot_alt_vol","tot_unalt_vol"],[total_altered,total_unaltered,tot_alt, tot_unalt]]
        values = pd.DataFrame(headers)
        print("---------------------+++++-------")
        print(values)
        print("---------------------+++++-------")
        values.to_csv(file[:-4]+"calculation.csv")
        #
       #
     #
    #
    ax1.plot(ratios,effective_pl_strain_alt, "or",ms=25)
    ax2.plot(ratios,effective_pl_strain_unalt,"ob",ms=25)
    ax11.plot(ratios,effective_pl_strain_alt, "or",ms=25)
    ax22.plot(ratios,effective_pl_strain_unalt,"ob",ms=25)
    #
    ax3.plot(ratios,effective_pl_strain_unalt,"db",ms=15)
    ax3.plot(ratios,effective_pl_strain_alt, "*r",ms=30)
    #
    ax5.plot([1]+ratios,vol_eff_pl_strain_unalt,"db",ms=15)
    ax5.plot([1]+ratios,vol_eff_pl_strain_alt, "*r",ms=30)
    #
    # Save values
    eff_str_vs_ratio = pd.DataFrame([[1]+ratios,vol_eff_pl_strain_alt,vol_eff_pl_strain_unalt])
    eff_str_vs_ratio.to_csv(target_dir+"sim_"+domain+"_"+str(sim_start))
    #
    print("target_dir,",target_dir)
    ax1.set_ylim([0,6])
    ax2.set_ylim([0,6])
    ax3.set_ylim([0,6])
    ax5.set_ylim([0,6])
    lineS="dotted"
    col="k"
    ax1.hlines(total_altered_iso/total_altered_iso,linestyle=lineS,color=col,xmin=1,xmax=4)
    ax11.hlines(total_altered_iso/total_altered_iso,linestyle=lineS,color=col,xmin=1,xmax=4)
    ax2.hlines(total_unaltered_iso/total_unaltered_iso,linestyle=lineS,color=col,xmin=1,xmax=4)
    ax22.hlines(total_unaltered_iso/total_unaltered_iso,linestyle=lineS,color=col,xmin=1,xmax=4)
    ax1.plot(1,total_altered_iso/total_altered_iso,"k*",ms=25)
    ax2.plot(1,total_unaltered_iso/total_unaltered_iso,"k*",ms=25)
    ax11.plot(1,total_altered_iso/total_altered_iso,"k*",ms=25)
    ax22.plot(1,total_unaltered_iso/total_unaltered_iso,"k*",ms=25)
    ax3.plot(1,total_altered_iso/total_altered_iso,"k*",ms=25)
    #
    ax3.plot(1,1,"k*",ms=25)
    #
    #ax1.set_title("$\sum \gamma^{\\alpha_{altered}}$")
    #ax2.set_title("$\sum \gamma^{\\alpha_{unaltered}}$")
    #ax11.set_title("$\sum \gamma^{\\alpha_{altered}}$")
    #ax22.set_title("$\sum \gamma^{\\alpha_{altered}}$")
    ax3.set_title("$\\bar\\varepsilon^{p}$ mesh calculation")
    ax5.set_title("$\\bar\\varepsilon^{p}$ Volume average")
    ax4.set_title("$\\bar\\varepsilon^{p}$")
    #comps = np.arange(12)
    #analysis_axes[-1].bar(comps,avg_slip_iso)
    #analysis_axes[-2].bar(comps,avg_slip_iso)
    #plt.tight_layout()
    if save_plot:

        df[str(sim_start)+domain+"_unaltered"]= [1]+effective_pl_strain_unalt
        df[str(sim_start)+domain+"_altered"]= [1]+effective_pl_strain_alt
        plt.savefig("/home/etmengiste/jobs/aps/images/Largeval_e"+simulations[sim_start]+"_"+domain,bbox_inches=mtransforms.Bbox([[.68, 0], [.85, 0.5]]).transformed(fig.transFigure - fig.dpi_scale_trans))
        print("###--##==+++plotted to file: Largeval_e"+simulations[sim_start]+"_"+domain)
    elif show:
        plt.tight_layout()
        plt.show()
    #
  #
#
#
## Plot effective plastic strain
def plot_eff_strain(start,all=False,marker_size=40,aps_home="/home/etmengiste/jobs/aps/eff_pl_str/"):
    y_label="$\\bar\\varepsilon^{p}$ (-)"
    for domain in ["Cube"]:

        if all:
            fig, axs = plt.subplots(1, 3,sharey="row")
            for j in range(3):
                ax = axs[j]
                ax.cla()
                ax.set_ylim([-0.1,2.1])
                ax.set_xlim([0.9,4.1])
                slip = slips[j]
                ax.set_title(slip+" slip systems strengthened")
                print(j+1)       
                for i in range(30*j,30*j+30,6):
                    print(i)
                    name=aps_home+"sim_"+domain+"_"+str(i)+"_eff_pl_str"
                    print(name)
                    dat = pd.read_csv(name,index_col=0)
                    #print(dat)
                    print("--")
                    ratios= dat.iloc[0]
                    altered= dat.iloc[1]
                    unaltered= dat.iloc[2]
                    print("altered ",altered)
                    print("unaltered ",unaltered)
                    set =int((i-30*j)/6)
                    print(set)
                    ax.plot(ratios,unaltered,"k-o",ls=sets[set],ms=marker_size,label="Set "+str(set+1))
                    ax.plot(ratios,altered,"kD",ls=sets[set],ms=marker_size)
                    ax.set_xticks(ratios)
                    ax.set_xticklabels(an,rotation=90)     
                    ax.set_yticks([0,0.5,1,1.5,2])
                    ax.set_yticklabels(["0.00","0.50","1.00","1.50","2.00"])  

            axs[0].set_ylabel(y_label,labelpad=25)
        else:
            fig= plt.figure()
            ax = fig.add_subplot(111)
            ax.cla()
            ax.set_ylim([0,2.1])
            ax.set_xlim([0.9,4.1])
            for i in range(start,start+25,5):
                dat = pd.read_csv(aps_home+"sim_"+domain+"_"+str(i)+"_eff_pl_str",index_col=0)
                #print(dat)
                print("--")
                ratios= dat.iloc[0]
                altered= dat.iloc[1]
                unaltered= dat.iloc[2]
                print(altered)
                print(unaltered)
                set =int((i-start)/5)
                ax.plot(ratios,unaltered,"k-*",ls=sets[set],ms=25,lw=4,label="Set "+str(set+1))
                ax.plot(ratios,altered,"D",ls=sets[set],ms=25,lw=4)
                ax.set_title(domain)
                ax.set_xticks(ratios)
                ax.set_xticklabels(an)

        sli = slips[int(start/25)]
        #ax.legend()
        y_label="$\\bar\\varepsilon^{p}$ (-)"


        fig.supxlabel(x_label,fontsize=SIZE)
        fig.subplots_adjust(left=0.09, right=0.98,top=0.9, bottom=0.2, wspace=0.07, hspace=0.1)
    
        plt.savefig("eff_pl_strain_"+sli+"ss"+domain)
    #plt.show()
  #
 #
#
##
##
def plot_yield_stress(start,all=False,marker_size=40):
    for domain in ["Cube", "Elongated"]:

        if all:
            fig, axs = plt.subplots(1, 3,sharey="row")
            for j in range(3):
                ax = axs[j]
                ax.cla()
                ax.set_ylim([127,175])
                slip = slips[j]
                ax.set_title(slip+" slip systems altered")
                print(j+1)       
                for i in range(25*j,25*j+25,5):
                    name=aps_home+"sim_"+domain+"_"+str(i)+"_yields"
                    dat = pd.read_csv(name,index_col=0)
                    #print(dat)
                    print("--")
                    ratios= dat.iloc[0]
                    yields = dat.iloc[1]
                    print(ratios)
                    print(yields)
                    set =int((i-25*j)/5)
                    ax.plot(ratios,yields,"ko",ls=sets[set],ms=marker_size,label="Set "+str(set+1))
                    ax.set_xticks(ratios)
                    ax.set_xticklabels(an,rotation=60)       
            axs[0].legend()
        else:   
            fig= plt.figure()
            ax = fig.add_subplot(111)
            ax.cla()
            ax.set_ylim([127,175])
            for i in range(start,start+25,5):
                dat = pd.read_csv(aps_home+"sim_"+domain+"_"+str(i)+"_yields",index_col=0)
                #print(dat)
                print("--")
                ratios= dat.iloc[0]
                yields = dat.iloc[1]
                print(ratios)
                print(yields)
                set =int((i-start)/5)
                ax.plot(ratios,yields,"ko",ls=sets[set],ms=15,label="SET "+str(set+1))

                ax.set_title(domain)
                ax.set_xticks(ratios)
                ax.set_xticklabels(an)
        sli = slips[int(start/25)]
        y_label="$\sigma_y$ (MPa)"
        

        fig.supxlabel(x_label)
        fig.supylabel(y_label)
        fig.subplots_adjust(left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1)
    
        plt.savefig("yield_stress"+sli+"ss"+domain)
    #plt.show()
    #
    print("------------------------------")
    #exit(0)
#
##
def get_bulk_output(simulation_plotted,domain):
    log_file= domain+"_"+simulation_plotted+".png"
    sim_path= home+"/"+simulation_plotted+"/"+domain
    os.chdir(sim_path)
    dot_sim =sim_path+".sim"
    sim=fepx_sim(simulation_plotted+domain,path=sim_path) # object created
    #sim.post_process("neper -S "+sim_path+"/ -reselset ori -resmesh stress,strain,slip,sliprate,crss")
    #if domain+".png" not in os.listdir(dot_sim):
    #    sim.post_process("~/code/scripts_such/ori_ipfs.sh")

    #if domain+"_density.png" not in os.listdir(dot_sim):
    #    sim.post_process("~/code/scripts_such/ori_ipfs_density.sh")

    #shutil.copyfile(dot_sim+"/"+domain+".png",simset_dir+"/common_files/figures/")
    #shutil.copyfile(dot_sim+"/"+domain+"_density.png",simset_dir+"/common_files/figures/")
    sim_results = sim.get_results()
    g_0= normalize(sim.material_parameters["g_0"],maximum=1)
    del sim
    #pprint(sim_results, preamble="#\n--------\n#")
    #
    fig = plt.figure(figsize=[60,60])
    #
    width = 5
    length = 2
    #
    plotted= 1
    #
    #
    #   Final Orientation density plot IPF
    #
    ax = fig.add_subplot(width,length,plotted)
    print(sim_path+".sim/")
    #im=plt.imread(os.getcwd()+".sim/"+domain+"_density.png")
    #ax.imshow(im)
    ax.set_axis_off()
    plotted+= 1
    #   Reorientation trajectories
    #
    ax = fig.add_subplot(width,length,plotted)
    print(sim_path+".sim/")
    #im=plt.imread(os.getcwd()+".sim/"+domain+".png")
    #ax.imshow(im)
    ax.set_axis_off()
    plotted+= 1
    #
    #
    #  Initial Slip strengths
    #
    ax = fig.add_subplot(width,length,plotted)
    plotting_space([g_0],axis=ax,ylabel="initial slip_strength",layers=2)
    plotted+= 1
    #
    #
    # Stress Strain
    #
    ax = fig.add_subplot(width,length,plotted)
    ax.set_xlabel("$\\varepsilon$")
    ax.set_ylabel("$\sigma$")
    component=2
    stress= normalize([i[component] for i in sim_results["stress"]],maximum=1)
    strain= normalize([i[component] for i in sim_results["strain"]],maximum=1)
    ax.plot(strain,stress,"k*-",linewidth=20)
    vals = find_yield(stress,strain)
    ax.plot(vals["y_strain"], vals["y_stress"], "o",label=vals["y_stress"])
    ax.legend(loc="best")
    ax.set_ylim([0,200])
    #
    #
    print("-------|---------|---------|---------|---------|---------|---------|---------|")
    print("-------|---------|---------|---------|---------|---------|---------|---------|")
    #
    for value in values:
        val=value.split("=")[1]
        try:
            steps_val= sim_results[val]
            #print(step,steps_val)
            steps_val = normalize(steps_val,maximum=1)
            plotted+=1
            ax = fig.add_subplot(width,length,plotted)
            if isinstance(steps_val,float):
                plotting_space([steps_val],axis=ax,ylabel=val,layers=2)
            elif isinstance(steps_val,list):
                plotting_space(steps_val,axis=ax,ylabel=val,layers=2)
        except:
            pass
        print("===+",plotted,"\n")
        #print("===+",plotted,"\n")
        #print("\n\n\n++++++",value,"\n",steps_val)
    #pprint(sim_results)
    #
    #os.chdir("../../")
    #
    fig.suptitle("simulations "+simulation_plotted)
    plt.tight_layout()
    #plt.show()
    plt.savefig("/home/etmengiste/jobs/aps/images/tex_output/"+domain+"_"+simulation_plotted)
    #
    #
  #
#
##
def get_yield_v_aniso_ratio(sim_start,domain,home=home,plot=False,ratios=[ 1.25, 1.5, 1.75, 2.0, 4],ids=[0],step = "28",res ="mesh"):
    simulations = os.listdir(home)
    simulations.sort()
    fig = plt.figure(figsize=[60,20])
    sim_iso = fepx_sim("iso",path=isotropic_home+"/Cube")
    num_steps = sim_iso.get_num_steps()
    iso_strain= [sim_iso.get_output("strain",step=step,res=res,ids=ids)[2] for step in range(num_steps)]
    iso_stress=[sim_iso.get_output("stress",step=step,res=res,ids=ids)[2] for step in range(num_steps)]
    #
    del sim_iso
    vals_iso = find_yield(iso_stress,iso_strain)
    ax= fig.add_subplot(131)
    ax.set_xlabel("$\\varepsilon$")
    ax.set_ylabel("$\sigma$")
    ax.set_ylim([0,250])
    #
    ax2= fig.add_subplot(132)
    ax2.set_xlabel("ratio")
    ax2.set_ylabel("$\sigma$")
    ax2.set_ylim([120,175])
    #
    ax3= fig.add_subplot(133)
    ax3.set_xlabel("ratio")
    ax3.set_ylabel("$|\sigma|$")
    ax3.set_ylim([0.99,1.4])
    #
    ax.plot(iso_strain,iso_stress,"k")
    ax.plot(vals_iso["y_strain"], vals_iso["y_stress"], "*k",label=vals_iso["y_stress"])
    ax2.plot(1,vals_iso["y_stress"], "*k",ms=20)
    ax3.plot(1,vals_iso["y_stress"]/vals_iso["y_stress"], "*k",ms=20)
    mrkr = ["ks","ko","kd","k^","kv"]
    yields = []
    #
    for index,sim_name in enumerate(simulations[sim_start:sim_start+5]):
        sim= fepx_sim(sim_name,path=home+sim_name+"/"+domain+".sim")
        num_steps = sim.get_num_steps()
        strain= [sim.get_output("strain",step=step,res="mesh")[2] for step in range(num_steps)]
        stress=[sim.get_output("stress",step=step,res="mesh")[2] for step in range(num_steps)]
        del sim
        ax.plot(strain,stress,"k.-")
        vals = find_yield(stress,strain)
        ax.plot(vals["y_strain"], vals["y_stress"], mrkr[index],ms=20,label=sim_name+": "+str(vals["y_stress"]))
        ax2.plot(ratios[index],vals["y_stress"], mrkr[index],ms=27)
        ax3.plot(ratios[index],vals["y_stress"]/vals_iso["y_stress"], mrkr[index],ms=27)
        yields.append(vals["y_stress"])
        ax.legend(loc=7)
        #plt.tight_layout()
    #
    yields = pd.DataFrame([[1]+ratios,[vals_iso["y_stress"]]+yields])
    yields.to_csv("/home/etmengiste/jobs/aps/sim_"+domain+"_"+str(sim_start)+"_yields")
    if plot:
        plt.savefig("/home/etmengiste/jobs/aps/aps_add_slip_ori/"+simulations[sim_start]+"_"+domain)
        print("\n+++--\n+++--\n+++-- wrote file "+simulations[sim_start]+"_"+domain)
        plt.clf()
    #
  #
#
##
def package_oris(path,name="elset_ori.csv"):
    sim= fepx_sim(path[-9],path=path)
    num_steps = sim.get_num_steps()+1
    del sim
    path = path+"results/elsets/ori/"
    steps = {"ori.step0": open(path+"ori.step0").readlines()}
    header =["Grain", "STEP_0_ROD_1", "STEP_0_ROD_2","STEP_0_ROD_3"]
    #
    for i in range(1,num_steps):
        #print("ori.step"+str(i))
        step= "ori.step"+str(i)
        steps[step]=open(path+step).readlines()
        header.append("STEP_"+str(i)+"_ROD_1")
        header.append("STEP_"+str(i)+"_ROD_2")
        header.append("STEP_"+str(i)+"_ROD_3")
        #print(steps[step][0:10])
    #
    values=[]
    #
    for i in range(2000):
        #print("Grain",i+1)
        grain_data= [i+1]
        for j in range(0,num_steps):
            #print(j,steps["ori.step"+str(j)][i].split()[0])
            grain_data.append(steps["ori.step"+str(j)][i].split()[0])
            grain_data.append(steps["ori.step"+str(j)][i].split()[1])
            grain_data.append(steps["ori.step"+str(j)][i].split()[2])
        values.append(grain_data)
    #
    arr= pd.DataFrame(values,columns=header)
    arr.to_csv("/home/etmengiste/jobs/aps/aps_add_slip_ori/"+name+".csv")
    print(" Wrote file "+name+".csv")
    #
  #
#
##
P = [ calculate_schmid(CUB_111[i],CUB_110[i]) for i in range(12)]
#
#
def calc_eff_pl_str(sim,domain,name="",under="",home=home,iso_home=home, debug=False):
    if name=="":
        name= domain
    file = open(home+sim+"/"+name+"_alt_eff_pl_str.csv","w")
    file_iso = open(iso_home+"isotropic"+"/"+name+"_eff_pl_str.csv","w")
    #
    sim= fepx_sim(sim,path=home+sim+"/"+domain)
    sim.post_process()
    num_elts = int(sim.sim['**general'].split()[2])
    step = sim.sim['**step']
    slip = sim.get_output("slip",step="28",res="elts",ids="all")
    elt_vol = sim.get_output("elt"+under+"vol",step="0",res="elts",ids="all")
    v_tot = sum(elt_vol[1]["0"])
    elt_vol_final = sim.get_output("elt"+under+"vol",step=step,res="elts",ids="all")
    mat_par = sim.material_parameters["g_0"]
    del sim
    #   ISO
    sim_iso= fepx_sim("name",path=iso_home+"isotropic/"+domain)
    sim_iso.post_process()
    step_iso = sim_iso.sim['**step']
    slip_iso = sim_iso.get_output("slip",step="28",res="elts",ids="all")
    elt_vol_iso = sim_iso.get_output("eltvol",step="0",res="elts",ids="all")
    v_tot_iso = sum(elt_vol_iso[1]["0"])
    elt_vol_final_iso = sim_iso.get_output("eltvol",step=step_iso,res="elts",ids="all")
    baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])
    #stress_iso = [normalize(sim_iso.get_output("stress",step=step,res=res,ids=ids)[str(i)]) for i in ids]
    del sim_iso
    #
    strength = [ float(i.split("d")[0]) for i in mat_par]
    #exit(0)
    altered  =  []
    ratio=1
    for index,val in enumerate(strength):
        if val>baseline:
            altered.append(index)
            ratio = val/baseline
    print(ratio)
    #
    #
    avg_eff_pl_str_alt = []
    avg_eff_pl_str_unalt = []
    #
    avg_eff_pl_str_alt_iso = []
    avg_eff_pl_str_unalt_iso = []
    #
    print("***---***")
    #print(altered)
    #print(baseline)
    print("***---***")
    values = "elt_vol, tot_vol, vol_frac, eff_pl_alt, eff_pl_unalt, vol_eff_pl_alt, vol_eff_pl_unalt"
    file.write(values+"\n")
    file_iso.write(values+"\n")
    #
    for el in range(num_elts):
        total_altered = 0
        total_unaltered = 0
        total_altered_iso = 0
        total_unaltered_iso = 0
        #
        for i in range(12):
            schmid_val = P[i]
            shear_val = slip[0][str(el)][i]
            shear_val_iso = slip_iso[0][str(el)][i]
            if i in altered:
                total_altered+= schmid_val*shear_val
                total_altered_iso+= schmid_val*shear_val_iso
                if debug:
                    print("\n+++Schmid val")
                    pprint(schmid_val, preamble="+++")
                    print("\n\n+++===slip system shear\n===---", shear_val)
                    print("\n+++===Total resolved shear strain")
                    pprint(total_altered, preamble="+++===---")
                    print("-----------------------------------##-------##-------##")
                    print("-----------------------------------##-------##-------##")
                    print("-----------------------------------##-------##\n\n")
            #print("-----------------------------------##-------##-------##")
            #
            else:
                total_unaltered+= schmid_val*shear_val
                total_unaltered_iso+= schmid_val*shear_val_iso
                if debug:
                    print("\n+++Schmid val")
                    pprint(schmid_val, preamble="+++")
                    print("\n\n+++===slip system shear\n===---", shear_val)
                    print("\n+++===Total resolved shear strain")
                    pprint(total_unaltered, preamble="+++===---")
                    print("-----------------------------------##-------##-------##")
                    print("-----------------------------------##-------##-------##")
                    print("-----------------------------------##-------##\n\n\n")
        #
        eff_pl_str_alt = math.sqrt((2/3)*inner_prod(total_altered,total_altered))
        eff_pl_str_unalt = math.sqrt((2/3)*inner_prod(total_unaltered,total_unaltered))
        #
        eff_pl_str_alt_iso = math.sqrt((2/3)*inner_prod(total_altered_iso,total_altered_iso))
        eff_pl_str_unalt_iso = math.sqrt((2/3)*inner_prod(total_unaltered_iso,total_unaltered_iso))
        #
        #
        v_el = elt_vol_final[0][str(el)][0]
        v_frac = v_el/v_tot
        #
        #iso
        v_el_iso = elt_vol_final_iso[0][str(el)][0]
        v_frac_iso = v_el_iso/v_tot_iso
        #
        # altered
        avg_eff_pl_str_alt.append(eff_pl_str_alt*v_frac)
        avg_eff_pl_str_unalt.append(eff_pl_str_unalt*v_frac)
        #
        #iso
        avg_eff_pl_str_alt_iso.append(eff_pl_str_alt_iso*v_frac_iso)
        avg_eff_pl_str_unalt_iso.append(eff_pl_str_unalt_iso*v_frac_iso)
        #
        if debug:
            print("el vol", v_el)
            print("tot vol", v_tot)
            print("vol_frac", v_frac)
            print("-----------------------------------##-------##\n")
            print("\n Effective plastic altered :",eff_pl_str_alt)
            print("\n Effective plastic altered :",eff_pl_str_unalt)
            print("-----------------------------------##-------##\n\n")
            print("-----------------------------------##-------##\n")
            print("\n Vol avg Effective plastic altered :",avg_eff_pl_str_alt[el])
            print("\n Vol avg Effective plastic altered :",avg_eff_pl_str_unalt[el])
            print("-----------------------------------##-------##\n\n")
        #
        values = str(v_el)+"," + str(v_tot)+","+ str(v_frac)+","+ str(eff_pl_str_alt)+ ","+ str(eff_pl_str_unalt)+ ","+ str(avg_eff_pl_str_alt[el])+ ","+ str(avg_eff_pl_str_unalt[el])
        file.write(values+"\n")
        #
        values = str(v_el_iso)+"," + str(v_tot_iso)+","+ str(v_frac_iso)+","+ str(eff_pl_str_alt_iso)+ ","+ str(eff_pl_str_unalt_iso)+ ","+ str(avg_eff_pl_str_alt_iso[el])+ ","+ str(avg_eff_pl_str_unalt_iso[el])
        #print(values)
        file_iso.write(values+"\n")
    file.close()
    file_iso.close()
    print("\n__")
    print(sum(avg_eff_pl_str_alt))
    print(sum(avg_eff_pl_str_unalt))
    file.close()
    file_iso.close()
  #
 #
#
##
def plot_svs_from_csv(name):
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 65})
    #plt.rcParams['text.usetex'] = True
    plt.rcParams['font.family'] = 'DejaVu Serif'
    plt.rcParams["mathtext.fontset"] = "cm"
    plt.rcParams['figure.figsize'] = 40,20
    ax  = plt.subplot()
    exx= pd.read_csv(name)
    stress_exp = [float(i) for i in exx.iloc[:,2]]
    strain_exp = [float(i) for i in exx.iloc[:,1]]
    nam=name.replace("_",' ')
    sampling_max = 125
    yield_values = find_yield(stress_exp,strain_exp,number=sampling_max)

    ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
    pprint(yield_values)
    ax.plot(strain_exp, stress_exp,"k-",markersize=2, lw=0.5)
    ax.plot(ystrain,ystress,"k*",ms=50,label="$\sigma_y$="+str(yield_values["y_stress"]))
    ax.hlines(sampling_max,xmin=0,xmax=0.02)
    #ax.scatter(strain_exp[0:len(strain_exp):name[4]], stress_exp[0:len(stress_exp):name[4]]
    #    , marker=yeild_markers[i],s=60,fc='w',color= 'k',zorder=i+3, label=nam[0:3])
    stress = '$\sigma$'
    strain='$\epsilon$'
    x_label = f'{strain} (-)'
    y_label = f'{stress} (MPa)'
    ax.legend(loc="lower right")
    # Compile labels for the graphs
    ax.set_ylim([0,270])
    ax.set_xlim([0,0.06])
    #lines_labels = [a.get_legend_handles_labels() for a in plt.axes]
    #lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
    #plt.legend(lines, labels,loc="best", fontsize="small")
    #plt.title(title,loc='center')
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    plt.tight_layout()
    file_name =name[:-4]
    plt.savefig(file_name+"_stress_v_strain.png")
    ax.cla()
#
##
def combined_ipf(arr):
    unirr_insitu_ff = "/home/etmengiste/jobs/aps/2023/Fe9Cr HEDM data repository/Fe9Cr-61116 (unirr)/in-situ ff-HEDM/"
    irr_insitu_ff1 = "/home/etmengiste/jobs/aps/2023/Fe9Cr HEDM data repository/KGT1119 (450C, 0.1 dpa)/In-situ ff HEDM/"
    irr_insitu_ff2 = "/home/etmengiste/jobs/aps/2023/Fe9Cr HEDM data repository/KGT1147 (300C, 0.1 dpa)/In-situ ff HEDM/"

    preamble= "In-situ FF Parent Grain Data "

    #
    #
    # Generates the ori files in
    for dirs in [unirr_insitu_ff, irr_insitu_ff1, irr_insitu_ff2]:
    #for dirs in arr:
        os.chdir(dirs)
        dir = [i for i in os.listdir(dirs) if i.endswith(".csv") and i.startswith(preamble)]
        dir.sort()
        # initial
        file = open(dirs+"ini","w")
        csv = pd.read_csv(dirs+dir[0])
        length = len(csv)
        for i in range(length):
            #print(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i]))
            file.write(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i])+"\n")
        # final
        file = open(dirs+"fin","w")
        csv = pd.read_csv(dirs+dir[-1])
        length = len(csv)
        for i in range(length):
            #print(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i]))
            file.write(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i])+"\n")
        # all
        file = open(dirs+"all","w")
        for i in dir:
                print(dirs+i[:-4].replace(" ","_")+"_ori")
                csv = pd.read_csv(dirs+i)
                length = len(csv)
                for i in range(length):
                    #print(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i]))
                    file.write(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i])+"\n")
        os.system("~/code/data_reduction_scripts/plot_ipf.sh")
#
##
def get_stress_strain(path,dir="z1",strain_rate = 1e-3):
    # Get the stress and strain for a given set of simulation
    #
    file = open(path+"post.force."+dir).readlines()[2:]
    stress = []
    strain = []
    for i in file:
        arr = i.split()[-3:]
        #print("string version",arr)

        arr = [float(n)  for n in arr]
        #print("float",arr)
        stress.append(arr[0]/arr[1])
        strain.append(arr[2]*strain_rate)
    return [stress,strain]
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
##
def quat_prod(q1, q2,debug=False):
       # Quaternion Product
       # input a q1 and 4*1
       many =False
       try:
              a = np.array([q[0] for q in q1.T])
              b = np.array([q[0] for q in q2.T])
              avect= np.array([q[1:] for q in q1.T])
              bvect= np.array([q[1:] for q in q2.T])
              if debug:
                     print("a",avect.shape)
                     print("b",bvect.shape)
              many=True
       except:
              a=q1[0]
              b=q2[0]
              avect=q1[1:]
              bvect=q2[1:]
       #
       a3 = np.tile(a,[3,1]).T
       b3 = np.tile(b,[3,1]).T

       if debug:
              print("a3",a3.shape)
       #
       if many:
              dotted_val = np.array([np.dot(a,b) for a,b in zip(avect,bvect)])
              crossed_val = np.array([np.cross(a,b) for a,b in zip(avect,bvect)])
       else:
              dotted_val=np.dot(avect,bvect)
              crossed_val=np.cross(avect, bvect)
       
       #
       ind1 =a*b - dotted_val
       if debug:
              print(ind1.shape)

       #print(crossed_val.shape)
       #print((b3*avect).shape)
       val=a3*bvect + b3*avect +crossed_val

       if debug:
              print(val)
       quat = np.array([[i,v[0],v[1],v[2]] for i,v in zip(ind1, val)])
       #
       if many:
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
       else:
              if quat[0]<0:
                     quat=-1*quat
              #print(quat)
              return quat
#
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
def ret_to_funda(quat, sym_operators=Cubic_sym_quats(),debug=False):
       #    Return quaternion to the fundamental region given symerty 
       #        operatiors
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
              pprint(sym_operators)
              print(sym_operators.shape)
       equiv_quats = quat_prod(reshaped_quat,np.tile(sym_operators,(1,n)))
       #pprint(equiv_quats)
       #exit(0)
       return equiv_quats
#
##
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
#
##
def quat_to_rod(val):
       # Quaternion to Rodrigues vector
       #
       omega = 2*math.acos(val[0])
       n = val[1:]/math.sin(omega/2)
       r = math.tan(omega/2)*n
       return r
#
#   Encorporate into fepx_sim class
def get_elt_ids(home,grain_id,domain="Cube"):
       file = open(home+"common_files/"+domain+".stelt").readlines()
       elt_ids = [i for i,line in enumerate(file) if line.split()[1]==str(grain_id)]
       return elt_ids
#
##
def rot_mat(arr1,arr2):
    #   Find the roation matrix from a basis matrix 
    #       Q_ij = arr1 => arr2
    #       Q_ji = arr1 => arr2
    Q_ij = []
    Q_ji = []
    if len(arr1) ==len(arr2):
        for a in arr1:
            temp = []
            for b in arr2:
                    temp.append(np.dot(a,b))
            Q_ij.append(temp)
        for b in arr1:
            temp = []
            for a in arr2:
                    temp.append(np.dot(a,b))
            Q_ji.append(temp)                     
        return [np.array(Q_ij),np.array(Q_ji)] 
    else:  
            print("not same size")
#
##
def dif_degs(start,fin,debug=False):
    q_ij,q_ji = rot_mat(start,fin)
    print(np.dot(q_ij.T,start[0]))
    print("---")
    print(np.dot(q_ij.T,fin[0]))
    v1 = normalize_vector(R.from_matrix(q_ij).as_quat())
    r1 = ret_to_funda(v1)
    pi=math.pi
    thet_ij =math.acos(min([r1[0],1]))*(180/pi)
    if debug:
        r2 = ret_to_funda(R.from_matrix(q_ji).as_quat())
        thet_ji =math.acos(r2[0])*(180/pi)
        return [thet_ij,thet_ji]
    
    return thet_ij
#
def eucledian_distance(arr1,arr2):
    sum = 0
    for a,b in zip(arr1,arr2):
        sum+=(a-b)**2
    return sum**0.5
##
def plot_rod_outline(ax):
    #
    a = [ (2**0.5)-1,   3-(2*(2**0.5)),  ((2**0.5)-1)]
    b = [ (2**0.5)-1,     ((2**0.5)-1), 3-(2*(2**0.5))]
    c = [ 3-(2*(2**0.5)),   (2**0.5)-1,   ((2**0.5)-1)]
    #
    X= [a[0],b[0],c[0]]
    Y= [a[1],b[1],c[1]]
    Z= [a[2],b[2],c[2]]

    #print("x",X)
    #print("y",Y)
    #print("z",Z)

    neg_x= [-i for i in X]
    neg_y= [-i for i in Y]
    neg_z= [-i for i in Z]

    X+=neg_x+X    +X    +neg_x+X    +neg_x+neg_x
    Y+=neg_y+Y    +neg_y+Y    +neg_y+neg_y+Y
    Z+=neg_z+neg_z+Z    +Z    +neg_z+Z    +neg_z


    # Plot the 3D surface
    ax.scatter(X,Y,Z,marker=".",c="k")
    #ax.plot_trisurf(X, Y, Z,alpha=0.3)
    ax.scatter(0, 0, 0,color="k")
    #https://stackoverflow.com/questions/67410270/how-to-draw-a-flat-3d-rectangle-in-matplotlib
    for i in range(0,len(X),3):
            verts = [list(zip(X[i:i+3],Y[i:i+3],Z[i:i+3]))]
            ax.add_collection3d(Poly3DCollection(verts,color="k",alpha=0.001))

    s = 3-(2*(2**0.5))
    l = (2**0.5)-1
    new_X= [l, l,  l,  l,  l,  l,  l, l]
    new_Y= [l, s, -s, -l, -l, -s,  s, l]
    new_Z= [s, l,  l,  s, -s, -l, -l, -s]
    vals = [new_X,new_Y,new_Z]
    for i in range(3):
            verts = [list(zip(vals[i-2],vals[i-1],vals[i]))]
            ax.add_collection3d(Poly3DCollection(verts,color="k",alpha=0.001))
    #
    #
    new_X= [-l, -l,  -l,  -l,  -l,  -l,  -l, -l]
    vals = [new_X,new_Y,new_Z]
    for i in range(3):
            verts = [list(zip(vals[i-2],vals[i-1],vals[i]))]
            ax.add_collection3d(Poly3DCollection(verts,color="grey",alpha=0.001))
    #
#
##
def plot_std_mean_data(NAME,ylims="",base=True,debug=False,**non_base):
    norm =False
    result ={}
    results = pd.read_csv(NAME+".csv").transpose()
    for i in results:
        result[results[i]["case"]]= [float(results[i]["mean"]),float(results[i]["std"])]
        #print(results[i]["case"],result[results[i]["case"]])
    if base==True:
        DOMAIN = [ "CUB"]
        DOM = [ "cubic"]
        NSLIP  = ["2","4", "6"]
        #NSLIP =["2"]
        ANISO  = ["125", "150", "175", "200", "400"]
        SETS    = ["1", "2", "3", "4", "5"]
        an = ["Iso.", "1.25", "1.50", "1.75", "2.00", "3.00", "4.00"]
        #SETS = ["1"]
        sets    =  ["solid","dotted","dashdot",(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))]
        #
        ###
        aniso = [100, 125, 150, 175, 200, 300, 400]
    else:
        ## finish later
        DOMAIN = non_base["DOMAIN"]
        DOM = non_base["DOM"]
        NSLIP  = non_base["NSLIP"]
        #NSLIP =["2"]
        ANISO  = non_base["ANISO"]
        SETS    = non_base["SETS"]
        an = non_base["an"]
        #SETS = ["1"]
        sets    =  non_base["sets"]
        #
        ###
        aniso = non_base["aniso"]
    differences= {}
    difs2={}

    if debug:
        NSLIP =["2"]
        DOMAIN = ["CUB"]
        DOM = ["cubic"]

    for dom in DOMAIN:
        fig, axs = plt.subplots(2, 3,sharex="col",sharey="row",figsize=(23,13))
        for slip in NSLIP:
            #
            ax0= axs[0][NSLIP.index(slip)]
            ax1= axs[1][NSLIP.index(slip)]
            #
            ax0.set_title(slip+" slip systems strengthened")
            #
            ax0.set_xlim([90,410])
            ax1.set_xlim([90,410])
            if ylims!="":
                ax0.set_ylim([0.95,2.0])
                ax1.set_ylim([0.95,2.0])
            else:
                ax0.set_ylim([-0.4,20.4])
                ax1.set_ylim([-0.4,10.4])
            #fig, ax = plt.subplots(1, 1,figsize=(10,12))
            for set,line in zip(SETS,sets):
                #
                try:
                    iso_mean = result["DOM_"+dom+"_ISO"][0]
                    iso_std =result["DOM_"+dom+"_ISO"][1]
                    norm = True
                except:
                    iso_mean= 0
                    iso_std = 0
                index1 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_125"]
                index2 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_150"]
                index3 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_175"]
                index4 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_200"]
                index5 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_300"]
                index6 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_400"]

                list1 = [iso_mean,index1[0], index2[0],
                    index3[0], index4[0], index5[0],index6[0]]
                list2 = [iso_std,index1[1], index2[1],
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
                ax1.plot(aniso,list2,"k",lw=2,linestyle=line,
                    label="Set "+str(set))
                #
                marker_size =130
                if set =="7":
                    ax0.scatter(iso_mean,
                        iso_mean, marker="o", s=marker_size,
                        edgecolor="k", label="$\sigma$ : Isotropic", color="k")
                    #
                    ax1.scatter(iso_std,
                        iso_std, marker="o", s=marker_size,
                        edgecolor="k", label="Isotropic", color="k")
                #
                for a, l, l2 in zip(aniso, list1, list2):
                    ax0.scatter(a, l, marker="o", s=marker_size,edgecolor="k",color= "k")
                    ax1.scatter(a, l2, marker="o", s=marker_size,edgecolor="k",color= "k")
                #
                ax0.set_xticks(aniso)
                ax0.set_xticklabels(an,rotation=90)

                ax1.set_xticks(aniso)
                ax1.set_xticklabels(an,rotation=90)
        if norm:
            val = "(-)"
        else:
             val= "($^{\circ}$)"
        axs[1][0].set_ylabel("$\\tilde{\\theta}$  "+val,labelpad=25)
        axs[0][0].set_ylabel("$\\bar{\\theta}$  "+val,labelpad=25)
        ###
        #
        # Put title here
        #title = "Deviation from Taylor Hypothesis: \n"+str(DOM[DOMAIN.index(dom)])+" domain, "+str(slip)+" slip systems\n"
        #
        deg = "$^{\circ}$"
        #
        x_label = f'$p$ (-)'
        y_label = f'Normalized Misorienation ({deg})'
        fig.supxlabel(x_label,fontsize=SIZE)
        plt.subplots_adjust(left=0.09, right=0.98,top=0.95, bottom=0.124, wspace=0.07, hspace=0.1)
        fig.savefig(NAME+str(DOM[DOMAIN.index(dom)])+"_mean_std.png",dpi=400)
#
#
#
def coordinate_axis(ax,ori,leng = 0.002,offset_text=1.6,offset= np.array([0.01,0,-0.0001]),
                    xyz_offset = [[-0.0005,-0.0007,0],[0,-0.001,0],[0,-0.001,-0.0004]],
                    sty = "solid",space="rod_real", fs=60):
    #
    #      defult params need to be fixed for each axis
    debug = True
    debug = False
    if space == "rod_real":
        rod_labs = ["$r_1,x$","$r_2,y$","$r_3,z$"]
    elif space== "real":        
        rod_labs = ["X","Y","Z"]
    axis_basis = [[1,0,0],[0,1,0],[0,0,1]]
    start = np.array(ori)+offset
    lab_offset = -0.0002
    ##
    ##     make into function
    ##
    for ind,eig in enumerate(axis_basis):
            lw= 4
            print(axis_basis)
            ax.quiver(start[0],start[1],start[2],
                    eig[0],eig[1],eig[2]
                    ,length=leng,normalize=True
                    ,color="k", linestyle = sty,linewidth=lw)
            #
            leng_text=offset_text*leng
            val_txt=(np.array(eig)*(leng_text))+np.array(start)+np.array(xyz_offset[ind])
            ax.text(val_txt[0],val_txt[1],val_txt[2], rod_labs[ind],fontsize=fs, ha='center',va='center',color='k')
            
            if debug:
                    start = np.array([0,0,0])
                    leng = 0.6
                    lab_offset = np.array([0.0005,0.00,0.0007])
                    lw= 5.6
                    val_txt = start(leng+lab_offset)
                    ax.text(val_txt[0],val_txt[1],val_txt[2], rod_labs[ind],fontsize=fs, color='k')
                    ax.quiver(start[0],start[1],start[2],
                        eig[0],eig[1],eig[2]
                        ,length=leng,normalize=True
                        ,color="k", linestyle = sty,linewidth=lw)
                    #
