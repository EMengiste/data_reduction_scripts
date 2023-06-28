import os
import shutil
from ezmethods import *

common_files="/home/etmengiste/jobs/aps/aps_add_slip/common_files"

os.chdir(common_files)

print(os.listdir())

completion_file= open("completed.txt","w")

os.chdir("..")
main=os.getcwd()

destination="/media/schmid_1tb_2/etmengiste/aps_add_slip/"
completed_sims= []
started_sims=[]



print(main,common_files)
sims=os.listdir()
domains = ["Cube"]
def pprint(arr):
    for a in arr:
        print(a)
debug=False
debug_move=False
#
completed="Info   : Final step terminated. Simulation completed successfully."
#
sim_suite_dir=os.getcwd()
post_process= False
#
if sim_suite_dir == destination:
    print("In the directory for analysis")
    post_process=True
#
print(sim_suite_dir)
list_of_sims= os.listdir(sim_suite_dir)
list_of_sims.sort()
list_of_sims.remove('common_files')
try:
    list_of_sims.remove("runtime.png")
    list_of_sims.remove("completed.txt")
except:
    pass
list_of_duplicates=[]
array=[]
# if debugging just go into the 001 folder
if debug:
    list_of_sims = ["001", "005", "015", "010"]
#
completion_file.write("   Domain:	Cubic,    Elongated")
# For all simulations in the sims directory
number = 0
for i in list_of_sims:
    # if the folder is not common_files
    print(i)
    completion_file.write("\n Sim_folder "+i+": ")
    print("Files in "+i,os.listdir(sim_suite_dir+"/"+i))
    sim_dir = sim_suite_dir+"/"+i
    # for the domain types available
    for j in domains:
        # debug with just the cube case
        if j in os.listdir(sim_dir):
            print("--"+j+ " available")
            domain = sim_dir+"/"+j
            #
            #print(os.listdir(domain))
            #
            # if the simulation for that domian was moved improperly fix that
            #
            if j in os.listdir(domain):
                print("what are you doing here")
                list_of_duplicates.append(domain)
                print(domain)

                for file in os.listdir(domain+"/"+j):
                    print(file)
                    try:
                        shutil.move(domain+"/"+j+"/"+file,domain)
                    except:
                        print("there's likely a double")
                        pass
            num=len(os.listdir(domain))
            if num>9:
                #
                out = [item for item in os.listdir(domain) if item.startswith("output.")]
                #
                out.sort()
                print(domain+"/"+out[1])
                output_file= open(domain+"/"+out[0],"r").readlines()
                #
<<<<<<< HEAD
                step="0"
                # Find out current step
                for out in output_file:
                    if out.startswith("Info   : Running step"):
                        step = out[17:]
                #
                output = output_file[-2]
                print(output)
                if output.startswith(completed):
                    print("Simulation completed----move")
                    # Show destination
                    print(destination+i+"/"+j)
                    completion_file.write("--Completed--")
                    #print(os.getcwd())
                    if debug_move:
                        print( "---"+destination+i+"/")
                        print(domain)
                        completed_sims.append(i+" "+j)
                    else:
                        if j in os.listdir(destination+i+"/"+j):
                            print("alread here bud")
                        else:
                            shutil.move(domain, destination+i+"/"+j)
                            print( " Moved to "+destination+i+"/"+j)
                        completed_sims.append(i+" "+j)
                        print("Completed "+completed_sims[-1])
                    if output == "_EVPS: Iterat":
                        pass
=======
                if j in os.listdir(domain):
                    print("what are you doing here")
                    list_of_duplicates.append(domain)
                    print(domain)

                    for file in os.listdir(domain+"/"+j):
                        print(file)
                        try:
                            shutil.move(domain+"/"+j+"/"+file,domain)
                        except:
                            print("there's likely a double")
                            pass
                num=len(os.listdir(domain))
                if num>9:
                    #
                    out = [item for item in os.listdir(domain) if item.startswith("output.")]
                    #
                    out.sort()
                    #print(domain+"/"+out[1])
                    output_file= open(domain+"/"+out[0],"r").readlines()
                    #
                    step="0"
                    # Find out current step
                    for out in output_file:
                        if out.startswith("Info   : Running step"):
                            step = out[17:]
                    #
                    output = output_file[-2]
                    print(output)
                    if output.startswith(completed):
                        print("Simulation completed----move")
                        # Show destination
                        print(destination+i+"/"+j)
                        completion_file.write("--Completed--")
                        #print(os.getcwd())
                        if debug_move:
                            print( "---"+destination+i+"/")
                            print(domain)
                            completed_sims.append(i+" "+j)
                        else:
                            if j in os.listdir(destination+i+"/"+j):
                                print("alread here bud")
                            else:
                                shutil.move(domain, destination+i+"/"+j)
                                print( " Moved to "+destination+i+"/"+j)
                            completed_sims.append(i+" "+j)
                            print("Completed "+completed_sims[-1])
                        if output == "_EVPS: Iterat":
                            pass
                        else:
                            array.append(float(output_file[-3][23:-6])/3600)
>>>>>>> 7b9bbbd011dc4c7f1974c1cfdfbc52ffdd8b49d7
                    else:
                        array.append(float(output_file[-3][23:-6])/3600)
                else:
                    completion_file.write("++Simulation started but not yet done++ currently on "+ step)
                    #  if simulation is not completed return current step
                    started_sims.append(i+" "+j+" currently on "+ step)
                    print("Simulation started but not yet done")

            if num==4:
                print("\n Simulation Initialized but not completed")
            else:
                pass
        else:
            print(j+" moved")
            completion_file.write("--Completed--")
            completed_sims.append(i+" "+j)

print("\n Incomplete sims are?\n")
pprint(started_sims)
print("\n\nSims completed are?")
pprint(completed_sims)
print("duplicates are in ")
pprint(list_of_duplicates)

pprint(array)
import matplotlib.pyplot as plt
import numpy as np

num= [i for i in range(len(array))]
plt.plot(num,array,"k*")
plt.axhline(np.mean(array),label="mean = "+str(round(np.mean(array),2))+" hrs")
plt.ylabel("time (hr)")
plt.xlabel("simulation number")
plt.legend()
#plt.show()
plt.savefig("runtime")
exit(0)
'''
        for j in domains:
            # debug with just the cube case
            if debug:
                j="Cube"

            # if the domain type is in the directory

            if j in os.listdir():
                # get into that directory and print the domain into the output file
                os.chdir(j)
                print("\t--Domain-- "+j+" ")
                completion_file.write("\t--Domain-- "+j+" ")

                #list of files in the current directory
                cwd_list= os.listdir()

                # If there are more than 9 files the simulaition is started
                if len(cwd_list)>9:
                    #
                    out = [item for item in os.listdir() if item.startswith("output.")]
                    #
                    print(out)
                    output_lines= open(out[0],"r").readlines()
                    output = output_lines[-2]
                    #
                    # Find out current step
                    for out in output_lines:
                        if out.startswith("Info   : Running step"):
                            step = out[17:]
                    #
                    # If the simulation Completion message is read
                    if output.startswith(completed):
                        print("Simulation completed----move")
                        # Show destination
                        print(destination+i+"/"+j)
                        completion_file.write("--Completed--")
                        #print(os.getcwd())
                        if debug_move:
                            print( "---"+destination+i+"/")
                            completed_sims.append(i+" "+j)
                        else:
                            shutil.move(os.getcwd(), destination+i+"/"+j)
                            completed_sims.append(i+" "+j)
                    else:
                        completion_file.write("  ++  Simulation started but not yet done  ++")
                        #  if simulation is not completed return current step
                        started_sims.append(i+" "+j+" currently on "+ step)
                        print("Simulation started but not yet done")

                    os.chdir("..")
                # If there are the exact amount of files needed to start the
                #  Simulation isnt started yet
                if len(cwd_list)== 4:
                    print("Initialized not started")
                    os.chdir("..")
            else:
                # if the simulation is already moved it is told as such
                print("Simulation Moved")
                for dom in domains:
                    completion_file.write("\n\t Completed--Domain-- "+dom+" ")
                    print(destination+i+"/"+dom)
                    print(os.getcwd()+"/"+dom)
                completion_file.write(" \n")
            os.chdir("..")
    else:
        pass
print("\n Incomplete sims are?\n")
pprint(started_sims)
print("\n\nSims completed are?")
pprint(completed_sims)

'''
