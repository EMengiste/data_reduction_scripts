import os
#
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