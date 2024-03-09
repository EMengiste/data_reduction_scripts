import os
import time
from datetime import datetime
import numpy as np
intro = 3*"++==++"+"||\n"
intro = 4*intro +"fepx_sim setup\n"+ 4*intro
print(intro," today is: ",datetime.now())
def pprint(arr):
    for i in arr:
        print(i)
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
    def __init__(self,name,path="",verbose=False):
        self.name=name
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
        self.sim_steps = ["0.0"]
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
        self.verbose = verbose
        self.is_sim=False
        self.results_dir=os.listdir(path)
        self.compressed=False
        # self.completed = "post.report" in self.results_dir
        self.has_config = "simulation.config" in self.results_dir or "simulation.cfg" in self.results_dir
        #
        if path[-4:] == ".sim":
            self.is_sim=True
            self.post_processed=True
            pass
        else:
            self.post_processed=os.path.isdir(path+".sim")
        #
        #
        #
        if verbose:
            print("-----------------------------------##")
            print("                   |||||------##")
            print("                   |||||------object < "+self.name+"> creation")
            print("                   |||||------##")
            print("---------------------------------##")
        else:
            path = "/".join(path.split("/")[-2:])
        #
        # If the config file exists poulate the attributes
        #
        if self.is_sim:
            print("########---- opening "+path+"/simulation.config")
            #self.post_process()
            try:
                config = open(self.path+"/inputs/simulation.config").readlines()
            except:
                config = open(self.path+"/inputs/simulation.cfg").readlines()
        elif self.has_config and not self.is_sim:
            #print(self.name,"has config")
            print("########---- opening "+path+"/simulation.config")
            config = open(self.path+"/simulation.config").readlines()
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
                config = open(self.path+"/simulation.config").readlines()
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
                if len(i.split())==2 and "Deformation History" not in i:
                    option= i.split()
                    self.deformation_history[option[0]]=option[1]
                #
                elif len(i.split())>3:
                    self.sim_steps.append(i.split()[1])
            if "Loading and Boundary Condition" in current:
                if len(i.split())==2 and "Deformation History" not in i:
                    option= i.split()
                    self.deformation_history[option[0]]=option[1]
                #
                elif len(i.split())>3:
                    self.sim_steps.append(i.split()[1])
            #
            if "Boundary Condition" in current:
                if len(i.split())>1 and "Boundary Condition" not in i :
                    option= i.split()
                    self.boundary_conditions[option[0]]=option[1]
                    #print(option)
                #
            if "Steps" in current:
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
        if self.verbose:
            self.get_summary()
        #
       #
      #
     #
    #
    #
    def get_num_steps(self):     
        file = open(self.path+"/.sim").readlines()
        num = file[-2]
        return int(num)
        exit(0)
        if self.deformation_history["number_of_steps"]!="":
            num = self.deformation_history["number_of_steps"]
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
    def get_mesh_stat(self):
        # print(self.path)
        outputs = [i for i in os.listdir(self.path) if i.startswith("output.")]
        # print(outputs)
        file= open(self.path+"/"+outputs[1]).readlines()
        start_ind=0
        for ind,i in enumerate(file):
            if i.startswith("Info   :   - Mesh parameters:"):
                start_ind = ind
                continue
        nodes,elts = int(file[start_ind+1].split(":")[-1]),int(file[start_ind+2].split(":")[-1])
        self.nodes = nodes
        self.elts = elts
        #
       #
      #
     #
    #
    #
    def get_runtime(self,unit="s"):
        outputs = [i for i in os.listdir(self.path) if i.startswith("output.")]
        try:
            file= open(self.path+"/"+outputs[1]).readlines()
            # print(file[-3].split(" "))
            self.runtime = float(file[-3].split(" ")[-2])
            # print(self.runtime)
        except:
            file= open(self.path+"/"+outputs[0]).readlines()
            # print(file[-3].split(" "))
            self.runtime = float(file[-3].split(" ")[-2])
            # print(self.runtime)
        if unit=="hr":
            self.runtime/=3600
        #
       #
      #
     #
    #
    #
    def get_results(self,steps=[],res="mesh",process=False):
        #
        # Available results
        print("____Results__availabe___are:")
        #
        def pprint(arr, preamble=""):
            for i in arr:
                i=str(i)
                if isinstance(arr,dict):
                    print(preamble+i+": "+str(arr[i])+"                   |||||------|")
                else:
                    print(preamble+i+"                   |||||------|")

        pprint(self.print_results, preamble="\n#__|")
        #
        node_only = ["coo","disp","vel"]
        mesh= [i for i in self.print_results if i not in node_only and i not in ["forces","ori"]]
        #
        results = str(mesh)[1:-1].replace("'","").replace(" ","")
        if process:
            self.post_process(options=f" -resmesh {results}")
        # exit(0)
        print("\n____Getting results at "+res+" scale\n   initializing results\n")
        #
        num_steps=self.get_num_steps()
        if res == "mesh":
            length= len(mesh)
        #
        results_dict= {"sim name":self.name ,"num steps": num_steps}
        #
        pprint(results_dict)
        #
        #
        for index in range(length):
            result=mesh[index]
            #print("\n\n--===== start"+result+"\n")
            steps =[]
            fill = '█'
            percent = round(index / float(length-1),3)
            filledLength = int(40 * percent)
            percent=int(percent*100)
            bar = fill * filledLength + '-' * (length - filledLength)
            prefix=" \n\n== Getting <"+res+">results for <"+result+">\n--step<"
            for step in range(num_steps):
                prefix+=str(step)
                if step==10:
                    prefix+="\n"
                try:
                    prefix=self.name+f" \n\n===== Getting <{res}> scale results for <{result}> at step <{step}>----\n-----<"
                    print(result,step)
                    vals = [float(i) for i in self.get_output(result,step=str(step),res=res)]
                    steps.append(vals)
                    print(f'\r{prefix} |{bar}| {percent}% ')
                except FileNotFoundError:

                    print(f"file <{result}> not found")
                    break
            prefix+= ">--------|\n+++\n+++"
            #print("--===== end"+result+"\n")
            results_dict[result]=steps
        return results_dict
        #
        #
    #
    #
    def get_output(self,output,id=0,step= "0", res="",ids=[0],num_steps=0,comp="",debug=False):
        step = str(step)
        value = {}
        ##
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
        #
        if output in ["coo","disp","vel"]:
            res ="nodes"
        elif res=="":
            res="mesh"
        #
        #
        # #    
        if self.path[-4:] == ".sim":
            step_file = self.path+"/results/"+res+"/"+output+"/"+output+".step"+step
        else:
            step_file = self.path+".sim/results/"+res+"/"+output+"/"+output+".step"+step
                #
        # print(step_file)
        # #      
        if step=="all":
            if num_steps==0:
                num_steps = self.get_num_steps()
            # print(num_steps)
            vals = []
            #
            # #    
            for step in range(num_steps):
                # print(str(step))
                value = self.get_output(output,ids=ids,step=step,res=res)
                if comp!="":
                    value = value[comp]
                vals.append(value)
            return np.array(vals)
        #
        # #    
        file = open(step_file)
        values=file.readlines()

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
        if options!="":
            print(options.split("-res"))
            print("\n\n")
            print(os.getcwd())
            print(options)
            os.chdir(self.path)
            os.system("neper -S . "+options)
            print("\n\n")
            with open(self.path+"/.sim") as file:
                self.sim=file.readlines()
                return
        #
        elif self.post_processed:
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
        print(self.name)
        pprint(self.material_parameters)
        pprint(self.deformation_history)
        pprint(self.boundary_conditions)
        print(self.completed)
        print(self.compressed)
        #
       #
      #
     #
    #
    def __repr__(self) -> str:
        return self.name
       #
      #
     #
    #
    #
    def __del__(self):
        if self.verbose:
            print("-----------------------------------##")
            print("                   |||||------##")
            print("                   |||||------object < "+self.name+"> destruction")
            print("                   |||||------##")
            print("---------------------------------##")
        else:
            pass
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


