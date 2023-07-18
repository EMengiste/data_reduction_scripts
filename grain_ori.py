import sys
import os
#sys.path.append("/home/etmengiste/code/data_reduction_scripts/")
from ezmethods import *

fractal=False
#fractal=True
rem=""
fract_name=""
home=rem+"/media/schmid_2tb_1/etmengiste/files/slip_system_study/"
homefull=rem+"/media/schmid_2tb_1/etmengiste/files/slip_system_study/"
if fractal:
    home=rem+"/media/schmid_1tb_2/etmengiste/aps_add_slip/"
    fract_name="_fractal_"
target_dir = "/home/etmengiste/jobs/aps/frag_grains_step_27/"
sample_num= 2000
start = 0
bin_size = 10
max = 0.00004
bins=np.arange(0,max,max/bin_size)
domains = [["Cube","CUB"], ["Elongated","ELONG"]]

os.chdir(home)
dirs = os.listdir()
dirs.sort()
print(dirs)
#exit(0)
grains = np.arange(0,2000,1)
#
ANISO  = ["1.25", "1.50", "1.75", "2.00", "3.00","4.00"]
slips = ["2","4","6"]
num =6
start_6ss = 90-6
step = "27"

final_step ="28"
##grains[28:29]
#["isotropic"]+dirs[num*4-num:num*4]+dirs[30+num*1-num:30+num*1]+
# 102,165,
for grain_id in [26,165,102]:
    elts = get_elt_ids(homefull,grain_id)
    #
    # ["isotropic"]+
    for sim_name in ["isotropic"]+dirs[30:37]:
        if sim_name=="common_files":
            print("common_files")
            break
        for dom in domains[:-1]:
            path = sim_name+"/"+dom[0]
            sim= fepx_sim("sim",path=home+path+".sim")
            sim.post_process()
            num_elts = int(sim.sim['general'].split()[2])
            nums= elts
            print("num elts ",len(nums))
            #
            ori_s = sim.get_output("ori",step="0",res="elts",ids=nums)
            ori_f = sim.get_output("ori",step=step,res="elts",ids=nums)
            #
            print("num oris step 0",len(ori_s))
            print("num oris step"+str(step),len(ori_f))
            #
            file_name = str(grain_id)+"_"+sim_name+"_oris"+fract_name
            file_s = open(target_dir+file_name+"0.txt","w")
            file_f= open(target_dir+file_name+str(step)+".txt","w")
            #os.system(n)
            for i in range(len(nums)):
                var=""
                var_f=""
                for j in range(3):
                    var+= str(ori_s[i][j])+" "
                    var_f+= str(ori_f[i][j])+" "
                #
                file_s.write(var+"\n")
                file_f.write(var_f+"\n")
            file_s.close()
            file_f.close()

            print(file_name)
            #
            #os.system("pwd") 
            os.chdir(target_dir)       
            #os.system("pwd") 
            #os.system("ls") 
            print(os.getcwd())
            # neper -V "Two(type=ori):file(${cu}0.txt),One(type=ori):file(${cu}28.txt)"
            print(sim_name)
            if sim_name!="isotropic":
                sim_num = int(sim_name)
                print(sim_num%6)
                aniso = sim_num%6-1
                num_slips = slips[int(sim_num/30)]
                #exit(0)
                print(sim_name)
                print(aniso)
                print(num_slips)
            #exit(0)
            os.system('sed -i "s/=FILE_NAME/='+file_name+'/g" batch_compiled_ori_ipfs.sh')             
            os.system("./batch_compiled_ori_ipfs.sh")
            os.system('sed -i "s/='+file_name+'/=FILE_NAME/g" batch_compiled_ori_ipfs.sh') 
            
            if sim_name!="isotropic":
                os.chdir("images")
                os.system("pwd")
                os.system('sed -i "s/=FILE_NAME/='+file_name+'/g" convert.sh') 
                #os.system('sed -i "s/[ANISO]/['+str(aniso)+']/g" convert.sh') 
                os.system('sed -i "s/=SLIPS/='+str(num_slips)+'/g" convert.sh')
                os.system("./convert.sh") 
                #os.system('sed -i "s/['+str(aniso)+']/[ANISO]/g" convert.sh')   
                os.system('sed -i "s/='+file_name+'/=FILE_NAME/g" convert.sh')  
                os.system('sed -i "s/='+str(num_slips)+'/=SLIPS/g" convert.sh')
            if sim_name=="isotropic":
                os.chdir("images")
                os.system("pwd")
                #os.system
            #exit(0)
            #exit(0)
   
exit(0)
neper_comand = ""
file_names = "neper -V \"final(type=ori):file("+sim_name+"_"+str(grain_id)+"oris"+str(final_step)+".txt),"
file_names += "start(type=ori):file("+sim_name+"_"+str(grain_id)+"oris0.txt)\""
neper_comand+= file_names+" -space ipf -datastartrad 0.020 -datastartcol white   -datafinalrad  0.007 black -datastartedgerad 0.01"
command = neper_comand+" -print "+str(grain_id)
print(command)
for ind,dir in enumerate(dirs[:-1]):
    path= home+"/"+dir+"/Cube.sim/"
    print(path)
    calc_eff_pl_str(path,"Cube",under="",home=home,iso_home=home, debug=False)


exit(0)