import math
import os
#
# helper functions
#
def pprint(arr,number=False): # Function for printing values in array line by line
    num=0
    for i in arr:
        print(str(num)+'. '+i)
        num+=1
        #
#
def splitter(arr):
    for i in range(len(arr)):
        arr[i] =arr[i].split()
#
def makeInt(n, t='i'): # helper function to convert list of strings to float values
    if t == 'i':
        for i in range(len(n)):
            n[i]= int(n[i])
    if t =='f':
        for i in range(len(n)):
            n[i]= float(n[i])
    return n
#
# For neper commands that require reading from file this creates the string
#   the descriptor
def fromFile(fileName,des=""):
    file = "\"file("+fileName+"_ori"
    if des!="":
        file+=",des="+des
    return(file+")\"")

#
# Outputs the line of the tesselation generating command
#
def tessellation(tess):
    fileName= tess["name"]
    num_grains = str(tess["num_grains"])
    morpho = tess["morpho"]
    reg= str(tess["reg"])
    tess_options= "neper -T -reg "+reg+" -n " + str(num_grains)+" -morpho "+morpho+" "
    try:
        if tess["ori_des"] != '':
            tess_options+='-ori '+fromFile(fileName,tess["ori_des"])
    except:
        pass
    return tess_options+" -o "+ fileName + "\n"
#
# Outputs the line of the mesh generating command
#
def mesh(mesh):
    tessName = mesh["name"]
    order= str(mesh["order"])
    rcl= str(mesh["rcl"])
    part= str(mesh["part"])
    tess_options= "neper -M " + tessName + ".tess -order " + order + " -rcl "+ rcl + " -part " + part
    return tess_options+" -o "+ tessName + "\n\n"
##
#
##  function to write the Material parameters
#
def writeMatParams(list, conf, unit= "" , phase_num= ''):
    numPhases = list['num_phases']
    #print(numPhases)
    conf.write("## Material Parameters\n\n")
    conf.write("#--Cij (MPa) \n\n")
    conf.write("    number_of_phases "+str(numPhases)+"\n\n")
    if unit == "":
        unit ="GPa"
    units = {'MPa': "",
             'GPa': "e3"}

    for i in range(1,numPhases+1):
        phase = "phase_"+str(i)
        conf.write("    "+phase+"\n\n")
        crystal_type= list[phase]
        conf.write("    crystal_type "+crystal_type+"\n")
        for key, value in list[phase+"_crys_slip"].items():
            conf.write(f"    {key} {value}\n")
        for key,value in list[phase+'_stiff'].items():
            conf.write(f"    {key} {value}"+units[unit]+"\n")
        if list["precip_hard"]=='True':
            for key,value in list[phase+'_precip_params'].items():
                conf.write(f"    {key} {value}\n")
        conf.write('\n')
#
##  function to write the boundary conditions
#
def writeBCs(list, conf):
    conf.write("## Boundary Condition\n\n")
    conf.write("    boundary_conditions "+list['boundary_conditions']+"\n\n")
    #
    #
    if list['boundary_conditions'] == 'uniaxial_grip':
        #
        conf.write("    loading_direction "+list["options"]["loading_direction"]
                    +"\n")
        conf.write("    loading_face "+list["options"]["loading_face"]
                    +"\n")
        #
    #
    #
    elif list['boundary_conditions'] == 'uniaxial_symmetry':
        #
        # write for uniaxial_symmetry
        pass
        #
    elif list['boundary_conditions'] == 'uniaxial_minimal':
        #
        # write for uniaxial_minimal
        pass
        #
    conf.write("    strain_rate "+list['strain_rate']+"\n\n")
#
##  function to write the print results
#
def writePrintOptions(list, conf):
    conf.write("## Printing Results\n")
    for i in list:
        key = i
        conf.write("    print "+key+"\n")
    conf.write("\n\n")
#
## here a list of the first 5 values in optionKeys is given to be written to file
## as a debugging step thorugh the flag debug_printing
#
#
#
def writeDefHist(list, conf):
    numSteps    = list['num_steps']
    start       = list['start']
    stop        = list['stop']
    increment   = list['increment'] # defaul 1 prints at each step
    inc = (stop-start)/numSteps
    conf.write("## Deformation History\n\n")
    conf.write("    def_control_by uniaxial_strain_target\n\n")
    conf.write("    number_of_strain_steps "+str(numSteps)+"\n\n")
    strainSteps = [round(i* inc,8) for i in range(numSteps)]
    if increment == 1:
        for i in strainSteps:
            conf.write("    target_strain "+str(i)+" 1 print_data\n")
    conf.write("\n\n")
#
#   --------------------
#   Generate basic config_file
#
#       write more to streamline itterable parts
#
def generate_config(samples,path):
    script = open("batch_post_process.sh","w")
    os.system("chmod 777 batch_post_process.sh")
    script.write("#!/bin/bash \n\n")
    for sample in samples:        
        if sample["name"]+".config" not in os.listdir(path):
            conf = open(sample["name"]+".config","w")
            writeMatParams(sample["mat_params"], conf)
            writeBCs(sample["BCs"], conf)
            writeDefHist(sample["defHist"], conf)
            writePrintOptions(sample["printOptions"], conf)
        script.write("\ncd "+sample["name"])
        script.write("\ncp ~/code/post_process.sh ./")
        script.write("\nsed 's/LOAD_STEP/"+str(sample["defHist"]['num_steps'])+"/' post_process.sh")
        script.write("\n ./post_process.sh\n")
        script.write("\ncd ../")
    script.write("\nexit 0")
    script.close()

#
# writes comands to prepare the simulations and prepare for batch run
#
def generate_mesh(samples, path, options= ""):
    # Create the generate_mesh script
    script = open("generate_mesh.sh","w")
    os.system("chmod 777 generate_mesh.sh")
    script.write("#!/bin/bash \n\n")
    if options == "":
        for sample in samples:
            if sample["name"]+".tess" not in os.listdir(path):
                script.write(tessellation(sample))
            if sample["name"]+".msh" not in os.listdir(path):
                script.write(mesh(sample))
            sample = sample["name"]
            script.write("\ncd ../")
            script.write("\nmkdir "+sample)
            script.write("\ncd "+sample)
            script.write("\ncp ../common_files/"+sample+".tess ./simulation.tess")
            script.write("\ncp ../common_files/"+sample+".msh ./simulation.msh")
            script.write("\ncp ../common_files/stdtriangle.png ./stdtriangle.png")
            script.write("\n../common_files/visualize.sh\n")
            script.write("\nmv simulation.png ../common_files/"+sample+".png")
            script.write("\nmv ../common_files/"+sample+".config ./simulation.config\n")
            script.write("\ncd ../common_files\n\n")

    else:
        pass #extend to have input tesselation and meshing options
    #
    script.write("\nexit 0")
    script.close()
    #
#
#
def batch_run(samples,path):
    script = open("batch_run.sh","w")
    os.system("chmod 777 batch_run.sh")
    script.write("#!/bin/bash \n\n")
    for sample in samples:
        num_partitions= sample["part"]
        script.write("\ncp slurm_run.sh ../"+sample["name"])
        script.write("\ncd ../"+sample["name"])
        script.write("\nsed 's/PART/"+str(num_partitions)+"/' slurm_run.sh")
        script.write("\nsbatch --hint=nomultithread slurm_run.sh")
    script.write("\nexit 0")
    script.close()
#
#
## default values for  basic  config file
mat_params = {'num_phases'    : 1,
           'phase_1'          : "FCC",
           'phase_1_crys_slip':{'m'          : "0.050d0",
                                'gammadot_0' : "1.0d0",
                                'h_0'        : "200.0d0",
                                'g_0'        : "210.0d0",
                                'g_s0'       : "330.0d0",
                                'n'          : "1.0d0"},
           'phase_1_stiff'      : { 'c11': "111.2e3",
                                    'c12': "57.4e3",
                                    'c44': "26.4e3"},
           'phase_1_precip_params': { 'a_p': "245.0",
                                      'f_p': "155.0",
                                      'b_p': "62.50",
                                      'c_p': "10",
                                      'b_m': "2.54e-1",
                                      'r_p': '33'},
           'precip_hard'    : True,
           }

BCs = {'boundary_conditions': 'uniaxial_minimal',
       'options'            : {"loading_direction": "Z",
                               "loading_face"     : "Z_MAX"},
       'strain_rate'        : "1e-2"}

defHist = {'num_steps': 20,
           'start'    : 0,
           'stop'     : 0.03,
           'increment': 1}




depo_params = {"name": "depo",
                "mat_params": mat_params,
                "BCs"      : BCs,
                "defHist"  : defHist,
                "printOptions": ["stress","strain","forces","coo","stress-eq","stress_eq"],
                "num_grains" : 2000,
                "morpho"     : "gg",
                "reg"        : 1,
                "order" : 2,
                "ori_des"   :"euler-bunge",
                "rcl"   : 1,
                "part"  : 48}

feed_params = {"name": "feed",
                "mat_params": mat_params,
                "BCs"      : BCs,
                "defHist"  : defHist,
                "printOptions": ["stress","strain","forces","coo","stress-eq","stress_eq"],
                "num_grains" : 20,
                "morpho"     : "gg",
                "reg"        : 1,
                "order" : 2,
                "ori_des"   :"euler-bunge",
                "rcl"   : 2,
                "part"  : 48}
samples = [depo_params , feed_params]

path = os.getcwd()
generate_config(samples,path)
generate_mesh(samples,path)
batch_run(samples,path)
