import sys
import os
import time
#sys.path.append("/home/etmengiste/code/data_reduction_scripts/")
from ezmethods import *

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
remote = "/run/user/1001/gvfs/sftp:host=acmelabpc2.eng.ua.edu,user=etmengiste"
#remote=""
home=remote+"/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_system_study/"
iso_home =remote+"/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"
for i in range(0,90,6):
    tic = time.perf_counter()
    slip_vs_aniso(i,"Cube.sim",slip_systems,base=6,iso_home=iso_home,target_dir=remote+"/home/etmengiste/jobs/aps/eff_pl_str/sim_",ratios=[ 1.25, 1.50, 1.75, 2.00,3.00, 4.0],path=home,debug=False,save_plot=False,res="elts")

    toc = time.perf_counter()
    print(f"calculated the eff pl strain for {dir} in {toc - tic:0.4f} seconds")
exit(0)
os.chdir(home)
dirs = os.listdir()
dirs.sort()
for ind,dir in enumerate(dirs[70:-1]):

    path= home+"/"+dir+"/Cube "
    print(path)    
    calc_eff_pl_str(dir,"Cube.sim",name="Cube",under="",home=home,iso_home=iso_home, debug=False)

    toc = time.perf_counter()
    print(f"calculated the eff pl strain for {dir} in {toc - tic:0.4f} seconds")
    
exit(0)
#calc_eff_pl_str(dir,"Cube",under="",home=home,iso_home=iso_home, debug=False)
slip_vs_aniso(0,"Cube.sim",slip_systems,path=home,debug=False,save_plot=False,res="elts")

exit(0)



#
remote = False
home=""
if remote:
    home="/run/user/1001/gvfs/sftp:host=acmelabpc2.eng.ua.edu,user=etmengiste"    
stelt_home="/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"
home+="/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"
target_dir = "/home/etmengiste/jobs/aps/files/"
sample_num= 2000
start = 0
bin_size = 10
max = 0.00004
bins=np.arange(0,max,max/bin_size)
domains = [["Cube","CUB"], ["Elongated","ELONG"]]

os.chdir(home)
dirs = os.listdir()
dirs.sort()
grains =  [102, 1502, 1961]
#
##grains[28:29]
sims = ["isotropic","001","002","003","004","005",
                    "026","027","028","029","030",
                    "051","052","053","054","055"]
#
for grain_id in grains:
    elts = []
    file = open(stelt_home+"common_files/Cube.stelt")
    for i in file.readlines():
        i= i.split()
        if i[1] == str(grain_id):
            elts.append(int(i[0])-1)
    file.close()
    #
    #
    for sim_name in sims:
        if sim_name=="common_files":
            print("common_files")
            break
        for dom in domains[:-1]:
            path = sim_name+"/"+dom[0]
            sim= fepx_sim("sim",path=home+path+".sim")
            sim.post_process()
            final_step = sim.get_num_steps()
            print(sim.sim['**general'])
            num_elts = int(sim.sim['**general'].split()[2])
            #exit(0)
            step = sim.sim['**step']
            nums= elts
            print("num elts ",len(nums))
            #
            ori_s = sim.get_output("ori",step="0",res="elts",ids=nums)
            ori_f = sim.get_output("ori",step=final_step,res="elts",ids=nums)
            #
            print("num oris step 0",len(ori_s))
            print("num oris step"+str(final_step),len(ori_f))
            #
            file_s = open(target_dir+sim_name+"_"+str(grain_id)+"oris0.txt","w")
            file_f= open(target_dir+sim_name+"_"+str(grain_id)+"oris"+str(final_step)+".txt","w")
            #os.system(n)
            for i in ori_s[0]:
                var=""
                var_f=""
                for j in range(3):
                    var+= str(ori_s[0][i][j])+" "
                    var_f+= str(ori_f[0][i][j])+" "
                #
                file_s.write(var+"\n")
                file_f.write(var_f+"\n")
            file_s.close()
            file_f.close()
            print(grain_id)
            #
            os.system("pwd") 
            os.chdir(home+sim_name)       
            os.system("pwd") 
            os.system("ls") 
            print(os.getcwd())
            #exit(0)
            #
            neper_comand = "neper -V"
            file_names = "final(type=ori):file(Cubeoris28.txt),start(type=ori):file(Cubeoris0.txt)"
            neper_comand+= " \""+file_names+"\" -space ipf -datastartrad 0.020 -datastartcol white   -datafinalrad  0.007 darkgrey -datafinaledgerad 0.01"
            command = neper_comand+" -print "+str(grain_id)
            print(command)
            #os.system(command)
            #exit(0)
            #exit(0)
   
exit(0)
for ind,dir in enumerate(dirs[:-1]):
    path= home+"/"+dir+"/Cube.sim/"
    print(path)
    calc_eff_pl_str(path,"Cube",under="",home=home,iso_home=home, debug=False)


exit(0)