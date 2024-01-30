import os
##   --------------------
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

######   Config file setup
#
##  function to write the material parameters
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
        if list['Consider Precip']:
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
    print_options = open("print_options",'r').readlines()
    printOptions = {}
    for i in print_options:
        lst = i.split(':')
        #print(lst)
        printOptions[lst[0]] = lst[1]
    # Option names in one list
    #
    optionKeys = list(printOptions.keys())
    conf.write("## Printing Results\n")
    for i in list:
        key = optionKeys[i]
        conf.write("    print "+key+"\n")
    conf.write("\n\n")
#
## here a list of the first 5 values in optionKeys is given to be written to file
## as a debugging step thorugh the flag debug_printing
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
#####  Neper mesh and tesselation setup
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
    mesh_name=tess_name
    commands = ["-part "+str(num_partition)]
    neper_command=source_code+" -M "+tess_name+" "+commands[0]
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
        neper_command=source_code+" -M -loadmesh "+mesh_name+".msh"   
        print("Meshing command:\n",neper_command) 
        if os.path.exists(mesh_name+".stelset"):
            return mesh_dir+"/"+mesh_name
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
def visualize(input_source,source_code="neper",overwrite_file=False,outname="default_img"
              ,options={"mode" :"run"}):
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
        if os.path.exists(outname+".png") and not overwrite_file:
            print("Image already exists",outname+".png")
        else:
            print("Image doesn't exist generating new")
            print(os.getcwd())
            os.system(neper_command+" > "+outname+"_output")#> vis_output"+input_name)
    elif options["mode"]=="rerun":
        os.system(neper_command+" > vis_output")
#
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
def write_crss_file(values,target_dir="",name="simulation",res="Elset",verbose=True):
    ####    Open crss file and write header
    #
    precip_dist_file = open(target_dir+name+".crss",'w')
    precip_dist_file.write("$"+res+"Crss\n")
    num_vals=len(values)
    precip_dist_file.write(str(num_vals)+" 1\n")
    for i in range(num_vals):
        precip_dist_file.write(str(i+1)+" "+str(values[i])+"\n")
    precip_dist_file.write("$End"+res+"Crss")
    if verbose:
        print("--===wrote file",target_dir+name+".crss")
#
#   Legacy code
# def generate_tess(n,name,main_dir=".",source_code="neper",options={"mode" :"run"}):
#     print("\n===")
#     curr_dir = os.getcwd()
#     os.chdir(main_dir)
#     tesselation= name

#     commands= ["-n "+str(n),"-o "+tesselation]
#     neper_command=source_code+" -T "+commands[0]
#     # populate the commands using the optional input dictionary
#     for i in options:
#         if i[0]== "-":
#             neper_command+=" "+i+' '+options[i]
#             commands.append(i+' '+options[i])
#     #
#     neper_command+=" "+commands[1]
#     if options["mode"] =="debug":
#         #print(source_dir)
#         print("-----debugging--------")
#         print("Tess command:\n",neper_command)
#         print("Commands list:")
#         pprint(commands)
#         print(os.getcwd())
#     elif options["mode"] =="run":
#         print("-----tesselation--------")
#         print("Tess command:\n",neper_command)
#         if os.path.exists(tesselation+".tess"):
#             print("tesselation already exists")
#         else:
#             print("tesselation doesn't exist generating new")
#             os.system(neper_command)

#     os.chdir(curr_dir)
#     return main_dir+"/"+tesselation+".tess"
# #
# def generate_msh(source_dir,num_partition,source_code="neper",options={"mode" :"run"}):
#     print("\n===")
#     curr_dir = os.getcwd()
#     input_name= source_dir.split("/")[:]
#     mesh_dir = "/".join(input_name[:-1])
#     os.chdir(mesh_dir)
#     tess_name =input_name[-1]
#     commands = ["-part "+str(num_partition)]
#     neper_command=source_code+" -M "+tess_name+" "+commands[0]
#     mesh_name=tess_name
#     # populate the commands using the optional input dictionary
#     for i in options:
#         if i[0]== "-":
#             neper_command+=" "+i+' '+options[i]
#             commands.append(i+' '+options[i])
#         if i== "-o":
#             mesh_name= options[i]

#     if options["mode"]=="debug":
#         #print(source_dir)
#         print("-----debugging--------")
#         print("Meshing command:\n",neper_command)
#         print("Commands list:")
#         pprint(commands)
#         print(os.getcwd())
#     elif options["mode"]=="run":
#         print("-----meshing--------")
#         print("Meshing command:\n",neper_command)
#         if os.path.exists(mesh_dir+"/"+mesh_name+".msh"):
#             print("Mesh already exists",mesh_dir+"/"+mesh_name+".msh")
#         else:
#             print("Mesh doesn't exist generating new")
#             os.system(neper_command+" > "+mesh_name+"_output")

#     elif options["mode"]=="stat":
#         print("-----getting_stats--------")
#         print("Meshing command:\n",neper_command)
#         neper_command=source_code+" -M -loadmesh "+mesh_name+".msh"            
#         for i in options:
#             if i.startswith("-stat"):
#                 neper_command+=" "+i+' '+options[i]
#         print(neper_command)
#         os.system(neper_command+" > "+mesh_name+"_output")

#     elif options["mode"]=="remesh":
#         print("-----remeshing--------")
#         neper_command=source_code+" -M -loadmesh "+mesh_name+".msh "+commands[0]
#         print("Meshing command:\n",neper_command)
#         os.system(neper_command+" > "+mesh_name+"_output")
#     os.chdir(curr_dir)
#     return mesh_dir+"/"+mesh_name+".msh"
# #
# def visualize(input_source,source_code="neper",options={"mode" :"run"}):
#     commands = []
#     input_name= input_source.split("/")[-1]
#     neper_command=source_code+" -V "+input_source+" "
#     # populate the commands using the optional input dictionary
#     for i in options:
#         if i[0]== "-":
#             neper_command+=" "+i+' '+options[i]
#             commands.append(i+' '+options[i])
#     neper_command+=" -print "+input_name[:-4]
#     if options["mode"]=="debug":
#         #print(input_source)
#         print(os.getcwd())
#         pprint(commands)
#         print("Visualization command:\n",neper_command)
#     elif options["mode"]=="run":
#         print("Visualization command:\n",neper_command)
#         if os.path.exists(input_source[0:-4]+".png"):
#             print("Image already exists",input_source[0:-4]+".png")
#         else:
#             print("Image doesn't exist generating new")
#             print(os.getcwd())
#             os.system(neper_command+" > vis_output"+input_name)
#     elif options["mode"]=="rerun":
#         os.system(neper_command+" > vis_output")
# #
# def post_process(sim_path,main_dir=".",options={"source code":"neper"}):
#     print("\n===")
#     print(sim_path)
#     neper_command= options["source code"]+" -S ."
#     if os.path.exists(main_dir+"/"+sim_path+".sim"):
#         print("Simulation folder exists")
#         print(sim_path+".sim")
#     else:
#         print("Simulation folder doesn't exist generating new")
#         print(sim_path)
#         if options["mode"]=="debug":
#             print("<debug_mode>")
#             os.chdir(main_dir+"/"+sim_path)
#             print(neper_command)
#             #
#             print(os.listdir())
#             #
#         elif options["mode"]=="run":
#             if os.path.exists("post.report"):
#                 print("Running post processing commands")
#                 os.system(neper_command)
#             else:
#                 print("simulation completed")
# #