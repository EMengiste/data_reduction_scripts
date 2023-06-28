import sys
import os
#sys.path.append("/home/etmengiste/code/data_reduction_scripts/")
from ezmethods import *


rem=""
home=rem+"/media/schmid_2tb_1/etmengiste/files/slip_system_study"
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
grains = np.arange(0,2000,1)
#
##grains[28:29]
#
for grain_id in [102, 1502, 1961]:
    elts = get_elt_ids(home+"/",grain_id)
    #
    print(elts)
    exit(0)
    #
    for sim_name in ["025","035"]:
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