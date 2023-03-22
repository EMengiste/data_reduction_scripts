import os
import shutil
common_files=os.getcwd()
print(os.listdir())
os.chdir("..")
main=os.getcwd()

destination="/media/ws3_2tb_1/files/slip_study_rerun/"

print(main,common_files)
sims=os.listdir()
domains = ["Cube","Elongated"]
def pprint(arr):
    for a in arr:
        print(a)
debug=False
completed="Info   : Final step terminated. Simulation completed successfully."
sim_suite_dir=os.getcwd()
print(sim_suite_dir)
list_of_sims= os.listdir(sim_suite_dir)
list_of_sims.sort()

array=[]
for i in list_of_sims:

    if i!= "common_files" and i!= "meta_data.txt":
        if debug:
            i="001"

        os.chdir(i)
        print(i)
        print(os.listdir())

        for j in domains:
            if debug:
                j="Cube"
            if j in os.listdir():
                os.chdir(j)
                print(j)
                cwd_list= os.listdir()
                if len(cwd_list)>9:
                    out = [item for item in os.listdir() if item.startswith("output.")]
                    print(out)
                    output= open(out[0],"r").readlines()[-3][23:-6]
                    print(i+" "+j+" "+output)
                    if output == "_EVPS: Iterat":
                        pass
                    else:
                        array.append(float(output))
                os.chdir("..")
        os.chdir("..")
    else:
        pass

import matplotlib.pyplot as plt
num= [i for i in range(150)]
plt.plot(num,array)
plt.show()
