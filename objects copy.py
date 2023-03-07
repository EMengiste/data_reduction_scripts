#
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
            if ids =="all":
                ids= [i for i in range(len(values))]
            for id in ids:
                value[str(id)]= [float(i) for i in values[id].split()]

            #pprint(value,max=1000)
        #pprint(dict)
        #print(value)
        if len(ids)==1:
            return value[str(ids[0])]
        else:
            return value
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
#
#____Results__availabe___are:
#
# Nodal outputs
#   coo             = [1,2,3]  x,  y,  z
#   disp            = [1,2,3] dx, dy, dz
#   vel             = [1,2,3] vx, vy, vz
# Element outputs (mesh,entity)
#   crss            = [1,2,...,n] where n=12,18,32 for bcc/fcc,hcp,bct respectively
#   defrate         = [1,2,3,4,5,6] tensor
#   defrate_eq      = [1]
#   defrate_pl      = [1,2,3,4,5,6]
#   defrate_pl_eq   = [1]
#   elt_vol         = [1]
#   ori             = [1,..,n] where n= 3 if rod of euler or 4 if quat or axis angle
#   slip            = [1,2,...,n] where n=12,18,32 for bcc/fcc,hcp,bct respectively
#   sliprate        = [1,2,...,n] where n=12,18,32 for bcc/fcc,hcp,bct respectively
#   spinrate        = [1,2,3] skew symetric plastic spin rate tensor
#   strain          = [1,2,3,4,5,6]
#   strain_eq       = [1]
#   strain_el       = [1,2,3,4,5,6]
#   strain_el_eq    = [1]
#   strain_pl       = [1,2,3,4,5,6]
#   strain_pl_eq    = [1]
#   stress          = [1,2,3,4,5,6]
#   stress_eq       = [1]
#   velgrad         = [1,2,3,4,5,6,7,8,9] full velocity gradient tensor
#   work            = [1]
#   work_pl         = [1]
#   workrate        = [1]
#   workrate_pl     = [1]
values = { "1=crss": [],
           "2=slip": [],
           "3=sliprate": [],
           "4=defrate": [],
           "5=defrate_eq": [],
           "6=defrate_pl": [],
           "7=defrate_pl_eq": [],
           "8=elt_vol": [],
           "9=ori": [],
           "10=spinrate": [],
           "11=strain": [],
           "12=strain_eq": [],
           "13=strain_el": [],
           "14=strain_el_eq": [],
           "15=strain_pl": [],
           "16=strain_pl_eq": [],
           "17=stress": [],
           "18=stress_eq": [],
           "19=velgrad": [],
           "20=work": [],
           "21=work_pl": [],
           "22=workrate": [],
           "23=workrate_pl": []}
########
########
########
import json
import os
from ezmethods import *
import matplotlib.pyplot as plt
plt.rcParams.update({'font.size': 10})
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams['figure.figsize'] = 12,12
import numpy as np
import shutil
import time
start= time.time()

def avg(arr):
    return sum(arr)/len(arr)
#
home="/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"
home="/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/"
home="/Users/ezramengiste/Documents/neper_fepx_gui/the_sims"


#sim.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
sim_iso= fepx_sim("name",path=home+"/1_uniaxial")
#sim_iso.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")

slip_systems = ["$(01-1)[111]$",                
                "$(10-1)[111]$",
                "$(1-10)[111]$",

                "$(011)[11-1]$",
                "$(101)[11-1]$",
                "$(1-10)[11-1]$",

                "$(011)[1-11]$",
                "$(10-1)[1-11]$",
                "$(110)[1-11]$",

                "$(01-1)[1-1-1]$",
                "$(101)[1-1-1]$",
                "$(110)[1-1-1]$"]
sim_iso.post_process()
val =sim_iso.sim["**general"].split()
sample_num = int(int(val[2])/10)
start = 0
bin_size = 10
ids = np.arange(start,start+sample_num,1)
step = "1"
res ="elts"
slip_iso =  [normailze(sim_iso.get_output("slip",step=step,res=res,ids=ids)[str(i)],absolute=True) for i in ids]
stress_iso = [normailze(sim_iso.get_output("stress",step=step,res=res,ids=ids)[str(i)]) for i in ids]
slips_iso = {}

baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])
del sim_iso
comps=np.arange(6)
fig = plt.figure(figsize=[60,20])
ax_stress = fig.add_subplot(2,7,13)
axies = []
for i in range(len(slip_iso[0])):
    slips =[]
    ax= fig.add_subplot(2,7,i+1)
    for id in ids:
        slips.append(slip_iso[id][i])
        if i == 0:
            stress_fake_1 =  [ 0.3*float(i) for i in stress_iso[id]] 
            plotting_space([stress_fake_1,stress_iso[id]],axis=ax_stress)
    slips_iso[str(i)] = slips
    #print(slips)
    slip_fake_1 = [  abs(float(i-0.002)) for i in slips] 
    ax.hist(slip_fake_1,bins=bin_size,color="red",edgecolor="k",alpha=0.2)
    ax.hist(slips,bins=bin_size,color="blue",edgecolor="k",alpha=0.2)
    ax.set_title(slip_systems[i])
    ax.set_ylim([0,int(sample_num)])
    axies.append(ax)


plt.tight_layout()
plt.subplots_adjust(left=0.062, bottom=0.08, right=0.9,top=0.948, wspace=0.162,hspace=.126)
plt.show()
exit(0)
for sim in ["046","050","075"]:
    sim= fepx_sim("name",path=home+sim+"/Cube")
    #sim.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
    slip =  [normailze(sim.get_output("slip",step="28",res="elsets",ids=ids)[str(i)],absolute=True) for i in ids]
    
    strength = [ float(i.split("d")[0]) for i in sim.material_parameters["g_0"]]
    altered  =  [index for index,val in enumerate(strength) if val>baseline]
    for i in range(len(slip_iso[0])):
        color = "k"
        if i in altered:
            color="red"
        slips =[]
        ax= axies[i]
        for id in ids:
            slips.append(slip_iso[id][i])
        slips_iso[str(i)] = slips
        ax.hist(slips,bins=20,color=color ,edgecolor="k", alpha= 0.2)
        ax.set_ylim([0,sample_num])

plt.tight_layout()
plt.subplots_adjust(left=0.13, right=0.995,top=0.99, bottom=0.1, wspace=0.035)
plt.savefig("figure")

exit(0)


baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])

sims = ["070","071","072","072","074", "075"]
slips_list = {}
for sim in sims:
    slips_list[sim] = slips

elt_num = len(ids)

del sim_iso

for simulation in sims:
    slips = slips_list[simulation]

    sim= fepx_sim("name",path=home+simulation+"/Cube")
    #sim.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
    slip =  [normailze(sim.get_output("slip",step="28",res="elsets",ids=ids)[str(i)],absolute=True) for i in ids]

    strength = [ float(i.split("d")[0]) for i in sim.material_parameters["g_0"]]
    altered  =  [index for index,val in enumerate(strength) if val>baseline]
    for j,val in enumerate(slip):
        ax= fig.add_subplot(2,6,j+1)
        slips[str(j)]=val 
        color="k"
        if j in altered:
            color="red"
        ax= fig.add_subplot(2,6,j+1)
        ax.hist(slips[str(j)], bins=20 ,color=color,edgecolor=color, alpha=0.5)       

        ax.set_ylim([0,1500])
    del sim
            
#stress = sim.get_output("stress",step="28",res="elsets",ids=ids)
#crss = sim.get_output("crss",step="28",res="elsets",ids=ids)

#pprint(slip,max=100)


#print(len(array_of_ids))
plt.tight_layout()
plt.subplots_adjust(left=0.13, right=0.995,top=0.99, bottom=0.1, wspace=0.035)
plt.show()
plt.savefig("figure")
exit(0)
array_of_ids = []
y = np.arange(6)
y2 = np.arange(12)
for index,i in enumerate(ids):
    if slips[index]> right and slips[index]<=left:
        #print(i,stress[str(i)])
        axs[1].bar(y, stress[str(i)],color="k",edgecolor="k",alpha=0.003)
        axs[2].bar(y2, slip[str(i)],color="k",edgecolor="k",alpha=0.003)
        axs[3].hist(crss[str(i)],color="k",edgecolor="k",alpha=0.003)
        array_of_ids.append(i)


stress = results["stress"]
strain = results["strain"]
