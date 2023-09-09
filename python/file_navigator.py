import os

def generate_tess(n,destination_name,source_dir,options={"mode" :"run"}):
    print("\n===")
    tesselation= main_dir+"/input_data"+destination_name

    commands= ["-n "+str(n),"-o "+tesselation]
    neper_comand=options["source code"]+" -T "+commands[0]
    # populate the comands using the optional input dictionary
    for i in options:
        if i[0]== "-":
            neper_comand+=" "+i+' '+options[i]
            commands.append(i+' '+options[i])
    #
    neper_comand+=" "+commands[1]
    if options["mode"] =="debug":
        #print(source_dir)
        print(os.getcwd())
        pprint(commands)
        print("Tess command:\n",neper_comand)
    elif options["mode"] =="run":
        print("Tess command:\n",neper_comand)
        if os.path.exists(tesselation+".tess"):
            print("tesselation already exists")
        else:
            print("tesselation doesn't exist generating new")
            os.system(neper_comand+' '+commands[1])

    #print(tess_destination+".tess")
    return tesselation+".tess"

def generate_msh(source_dir,num_partition,options={"mode" :"run"}):
    print("\n===")
    commands = ["-part "+str(num_partition)]
    neper_comand=options["source code"]+" -M "+source_dir+" "+commands[0]
    # populate the comands using the optional input dictionary
    for i in options:
        if i[0]== "-":
            neper_comand+=" "+i+' '+options[i]
            commands.append(i+' '+options[i])

    if options["mode"]=="debug":
        #print(source_dir)
        print(os.getcwd())
        pprint(commands)
        print("Meshing command:\n",neper_comand)
    elif options["mode"]=="run":
        print("Meshing command:\n",neper_comand)
        if os.path.exists(source_dir[0:-5]+".msh"):
            print("Mesh already exists",source_dir[0:-5]+".msh")
        else:
            print("Mesh doesn't exist generating new")
            os.system(neper_comand)
    elif options["mode"]=="remesh":
        os.system(neper_comand)
#
#
def post_process(sim_path,options={"source code":"fepx"}):
    print("\n===")
    print(sim_path)
    neper_comand= options["source code"]+" -S ."
    if os.path.exists(main_dir+"/"+sim_path+".sim"):
        print("Simulation folder exists")
        print(sim_path+".sim")
    else:
        print("Simulation folder doesn't exist generating new")
        print(sim_path)
        if options["mode"]=="debug":
            print("<debug_mode>")
            os.chdir(main_dir+"/"+sim_path)
            print(neper_comand)
            #
            print(os.listdir())
            #
        elif options["mode"]=="run":
            if os.path.exists("post.report"):
                print("Running post processing comands")
                os.system(neper_comand)
            else:
                print("simulation completed")
#
def run_sim(path_to_msh,path_to_config,sim_path,options={"source code":"fepx"}):
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
        os.system(neper_comand)
#
#
def pprint(arr):
    for i in arr:
        print("+=>",i)
#
print("1 ", os.getcwd())
os.chdir(path="..")
main_dir=os.getcwd()
print("2 ",os.getcwd())
#
#   print the current working direcotry and list the
#
folders= os.listdir()
pprint(folders)
#
#
# add functionality to take direct ff data
experimental_source="path/to/experimental/data"
simulation_tess="/default/simulation"
options ={"morpho":  "gg",
          "mode" : "debug",
          "source code" : "neper"}

sim= generate_tess(50,simulation_tess,main_dir,options)
# tess above has destination sim
#
options ={"-order":"2",
          "mode" :"debug",
          "source code":"neper"}
generate_msh(sim,4, options)
os.chdir(main_dir)
#

options ={"mode" :"debug",
          "cores": 4,
          "source code":"fepx"}
destination= "output_data/default/"
print(main_dir+destination+"000")
run_sim(sim[:-4]+"msh", sim[:-4]+"config", destination+"000", options)


options ={"mode" :"debug",
          "source code":"neper"}
post_process(destination+"000",options=options)
# Work on config file and run sim
