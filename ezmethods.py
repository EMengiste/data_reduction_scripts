import os
import numpy as np
import json
import matplotlib.pyplot as plt
import math
import pandas as pd
plt.rcParams.update({'font.size': 25})
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.subplot.left"] = 0.045
plt.rcParams["figure.subplot.bottom"] = 0.08
plt.rcParams["figure.subplot.right"] = 0.995
plt.rcParams["figure.subplot.top"] = 0.891
plt.rcParams["figure.subplot.wspace"] = 0.21
plt.rcParams["figure.subplot.hspace"] = 0.44
plt.rcParams['figure.figsize'] = 60,20
import numpy as np
home ="/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/"
ist= "/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/isotropic"


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
        #
        self.name=name
        #
        self.path=path
        #
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
        self.results_dir=os.listdir(path)
        self.completed = "post.report" in self.results_dir
        self.has_config = "simulation.config" in self.results_dir
        self.post_processed=os.path.isdir(path+".sim")
        #
        #
        #
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
      #
     #
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
        step_file = self.path+".sim/results/"+res+"/"+output+"/"+output+".step"+step

        with open(step_file) as file:
            values=file.readlines()
            num_components= len(values[0].split(" "))
            component_vals = {}
            for component in range(num_components):
                component_vals[str(component)] = []
            if ids =="all":
                ids= [i for i in range(len(values))]
            for id in ids:
                value[str(id)]= [float(i) for i in values[id].split()]
                for component in range(num_components):
                    component_vals[str(component)].append(value[str(id)][component])
            #pprint(value,max=1000)
        if len(ids)==1:
            return value[str(ids[0])]
        else:
            return [value,component_vals]
        #
       #
      #
     #
    #
    #
    def post_process(self,options=""):
        #
        if not self.completed:
            print("simulation not done come back after it is")
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
        elif self.post_processed:
            print("Already post processed")
            if self.sim == "":
                values = {}
                with open(self.path+".sim/.sim") as file:
                    sim_file = [i.strip() for  i in file.readlines()]
                    for line in sim_file:
                        if line.startswith("***"):
                            print(line,"\n")
                        elif line.startswith("**"):
                            values[line]= sim_file[sim_file.index(line)+1].strip()
                self.sim= values
            pprint(values)
            print(values["**general"][8])
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

def inner_prod(a,b):
    tot = 0
    for i in range(3):
        for j in range(3):
            tot+= a[i][j]*b[i][j]            
    return tot

def nomalize_vector(vect):
    value= 0
    final = vect
    for i in vect:
        value+=(i**2)
    mag=math.sqrt(value)
    for i in range(len(vect)):
        final[i] = final[i]/mag
    return final

def calculate_schmid(a,b):
    a= nomalize_vector(a)
    b= nomalize_vector(b)
    T = diad(a,b)

    P = 0.5*(T+np.transpose(T))
    #vect = vectorize(matrix)
    return P
    #print(calculate_inner(vect,vect))


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
##
# yield calculation for simulation data
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
    	#
        	if  stress_off[i] > stress[i]:
        		x_n1 = i
        		x_n = x_n1 - 1
        		break
        # Use Cramer's rule to find where lines intersect (assume linear between
        #  simulation points)

        m1 = E
        a = - E *  offset

        m2 = (stress[x_n1] - stress[x_n]) / (strain[x_n1] - strain[x_n])
        b = stress[x_n] - (m2 * strain[x_n])


        Ystrain = (a - b) / (m2 - m1)
        Ystress = ((a * m2) - (b * m1)) / (m2 - m1)
    values ={"y_stress": Ystress,
             "y_strain": Ystrain,
             "stress_offset": stress_off}
    return values
  #
 #
#
# Plot effective plastic strain
def plot_eff_strain(j,ax,domain,ani,dataframe):
    for i in range(0+25*j,24+25*j,5):
        mk="*"
        ax.plot(ani,dataframe[str(i)+domain+"_unaltered"],"b-"+mk,ms=15)
        ax.plot(ani,dataframe[str(i)+domain+"_altered"],"r-"+mk,ms=15)
        ax.set_xlabel("ratio")
        ax.set_ylabel("$\\bar\\varepsilon^{p}$")
        ax.set_title(domain)
##
def avg(arr):
    return sum(arr)/len(arr)
  #
 #
#
##
def slip_vs_aniso(sim_start,domain,slip_systems,debug=False,save_plot=False,df="", ids=[0],ratios=[ 1.25, 1.5, 1.75, 2.0, 4],step = "28",res ="mesh"):
    
    P = [ calculate_schmid(CUB_111[i],CUB_110[i]) for i in range(12)]
    sim_iso= fepx_sim("name",path=home+"isotropic/"+domain)
    #sim_iso.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
    sim_iso.post_process()
    baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])
    val =sim_iso.sim["**general"].split()
    slip_iso = normalize(sim_iso.get_output("slip",step=step,res=res,ids=ids),absolute=True)
    baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])
    #stress_iso = [normalize(sim_iso.get_output("stress",step=step,res=res,ids=ids)[str(i)]) for i in ids]
    del sim_iso
    #for i in range(6):
        #ax_stress = fig.add_subplot(hig,wid,12+i)
        #ax_stress.hist(stress_iso[str(i)],bins=bin_size,edgecolor="k")
        #ax_stress.set_title("Stress "+str(i))
        #stress_axies.append(ax_stress)
    #avg_slip_iso = []
    fig = plt.figure(1,figsize=[70,35])
    wid=6
    hig =3
    fig.clear()
    axies = []
    mean_iso = {}
    ax1= fig.add_subplot(hig,wid,13)
    ax2= fig.add_subplot(hig,wid,15)
    ax11= fig.add_subplot(hig,wid,14)
    ax22= fig.add_subplot(hig,wid,16)
    ax3= fig.add_subplot(hig,wid,17)
    ax4= fig.add_subplot(hig,wid,18)
    ax1.set_ylim(0,0.05)
    ax1.set_xlim(0.9,4.1)
    ax2.set_ylim(0,0.3)
    ax2.set_xlim(0.9,4.1)

    ax1.set_xlabel("ratio")
    ax2.set_xlabel("ratio")
    ax1.set_ylabel("$\\bar\\varepsilon^{p,a}$ ")
    ax11.set_ylabel("$\\bar\\varepsilon^{p,a}$ ")
    ax2.set_ylabel("$\\bar\\varepsilon^{p,u}$")
    ax22.set_ylabel("$\\bar\\varepsilon^{p,u}$")
    ax3.set_ylabel("normalized $\\bar\\varepsilon^{p}$")
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

    simulations = os.listdir(home)
    simulations.sort()
    effective_pl_strain_unalt = []
    effective_pl_strain_alt = []
    for sim in simulations[sim_start:sim_start+5]:
        sim= fepx_sim(sim,path=home+sim+"/"+domain)
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

        total_altered=math.sqrt((2/3)*inner_prod(total_altered,total_altered))
        total_unaltered=math.sqrt((2/3)*inner_prod(total_unaltered,total_unaltered))
        total_altered_iso=math.sqrt((2/3)*inner_prod(total_altered_iso,total_altered_iso))
        total_unaltered_iso=math.sqrt((2/3)*inner_prod(total_unaltered_iso,total_unaltered_iso))

        if debug:
            print("\n\n\n\n++")
            print(total_unaltered)
            print(total_altered_iso)
            print(total_unaltered_iso)

        effective_pl_strain_alt.append(total_altered/total_altered_iso)
        effective_pl_strain_unalt.append(total_unaltered/total_unaltered_iso)
        #
       #
     #
    #
    ax1.plot(ratios,effective_pl_strain_alt, "or",ms=25)
    ax2.plot(ratios,effective_pl_strain_unalt,"ob",ms=25)
    ax11.plot(ratios,effective_pl_strain_alt, "or",ms=25)
    ax22.plot(ratios,effective_pl_strain_unalt,"ob",ms=25)
    ax3.plot(ratios,effective_pl_strain_unalt,"db",ms=15)
    ax3.plot(ratios,effective_pl_strain_alt, "*r",ms=30)
    ax1.set_ylim([0,6])
    ax2.set_ylim([0,6])
    ax3.set_ylim([0,6])
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

    #ax1.set_title("$\sum \gamma^{\\alpha_{altered}}$")
    #ax2.set_title("$\sum \gamma^{\\alpha_{unaltered}}$")
    #ax11.set_title("$\sum \gamma^{\\alpha_{altered}}$")
    #ax22.set_title("$\sum \gamma^{\\alpha_{altered}}$")
    ax3.set_title("$\\bar\\varepsilon^{p}$")
    ax4.set_title("$\\bar\\varepsilon^{p}$")
    #comps = np.arange(12)
    #analysis_axes[-1].bar(comps,avg_slip_iso)
    #analysis_axes[-2].bar(comps,avg_slip_iso)
    #plt.tight_layout()
    df[str(sim_start)+domain+"_unaltered"]= [1]+effective_pl_strain_unalt
    df[str(sim_start)+domain+"_altered"]= [1]+effective_pl_strain_alt
    if save_plot:
        plt.savefig("/home/etmengiste/jobs/aps/images/Largeval_e"+simulations[sim_start]+"_"+domain)
        print("###--##==+++plotted to file: Largeval_e"+simulations[sim_start]+"_"+domain)
    #
  #
#
#
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

    fig = plt.figure(figsize=[60,60])

    width = 5
    length = 2

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
#
#
##
def get_yield_v_aniso_ratio(sim_start,domain,ratios=[ 1.25, 1.5, 1.75, 2.0, 4],ids=[0],step = "28",res ="mesh"):
    simulations = os.listdir(home)
    simulations.sort()
    fig = plt.figure(figsize=[60,20])
    sim_iso = fepx_sim("iso",path=ist+"/Cube")
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
    #
    for index,sim_name in enumerate(simulations[sim_start:sim_start+5]):
        sim= fepx_sim(sim_name,path=home+sim_name+"/"+domain)
        num_steps = sim.get_num_steps()
        strain= [sim.get_output("strain",step=step,res="mesh")[2] for step in range(num_steps)]
        stress=[sim.get_output("stress",step=step,res="mesh")[2] for step in range(num_steps)]
        del sim
        ax.plot(strain,stress,"k.-")
        vals = find_yield(stress,strain)
        ax.plot(vals["y_strain"], vals["y_stress"], mrkr[index],ms=20,label=sim_name+": "+str(vals["y_stress"]))
        ax2.plot(ratios[index],vals["y_stress"], mrkr[index],ms=27)
        ax3.plot(ratios[index],vals["y_stress"]/vals_iso["y_stress"], mrkr[index],ms=27)
        ax.legend(loc=7)
        #plt.tight_layout()
    #
    plt.savefig("/home/etmengiste/jobs/aps/images/figure_"+simulations[sim_start]+"_"+domain)
    print("\n+++--\n+++--\n+++-- wrote file "+simulations[sim_start]+"_"+domain)
    plt.clf()
    #
  #
#
#
#
##
def package_oris(path,name="elset_ori.csv"):
    sim= fepx_sim("sim_name",path=path[:-5])
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

    arr= pd.DataFrame(values,columns=header)
    arr.to_csv("/home/etmengiste/jobs/aps/slip_system_anistropy_study/"+name+".csv")
    print(" Wrote file "+name+".csv")
    #
  #
#
#