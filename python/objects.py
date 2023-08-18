#
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
########
########
########
import os
from ezmethods import *
import matplotlib.pyplot as plt

import math
plt.rcParams.update({'font.size': 45})
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.subplot.left"] = 0.05
plt.rcParams["figure.subplot.bottom"] = 0.08
plt.rcParams["figure.subplot.right"] = 0.99
plt.rcParams["figure.subplot.top"] = 0.95
plt.rcParams["figure.subplot.wspace"] = 0.21
plt.rcParams["figure.subplot.hspace"] = 0.44
plt.rcParams['figure.figsize'] = 30,20
import numpy as np
import shutil
import time
import pandas as pd
import time
start = time.perf_counter()

#sim.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
#sim_iso= fepx_sim("name",path=home+"/1_uniaxial")
#sample_num = int(int(val[2])/10)
sample_num= 2000
start = 0
bin_size = 10
max = 0.00004
bins=np.arange(0,max,max/bin_size)
domains = [["Cube","CUB"], ["Elongated","ELONG"]]
#ids = np.arange(start,start+sample_num,1)
home ="/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/"
aps_home ="/home/etmengiste/jobs/aps/"
home="/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"
simulations = os.listdir(home)
simulations.sort()
bulk=False

slips = ["2","4","6"]
aniso = ["125","150","175","200","400"]
sets    =  ["solid","dotted","dashdot",(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))]

an = ["Iso.", "1.25", "1.50", "1.75", "2.00", "4.00"]
#name = "DOM_"+domains[0][1]+"_ISO"
#package_oris(home+"isotropic/"+domains[0][0]+".sim/", name=name)

file = open(home+"common_files/Cube.stelt").readlines()
grain_id = 300
file_grain_ids = open(home+"common_files/Cube_grain_elts")

elts = file_grain_ids.readlines()[grain_id].split(", ")
elts = [int(i) for i in elts]
file_grain_ids.close()
print(elts[0:5])
exit(0)
#name = "DOM_"+domains[1][1]+"_ISO"
#package_oris(home+"isotropic/"+domains[1][0]+".sim/", name=name)
for sim_name in simulations[0:5]:
    if sim_name=="common_files":
        print("common_files")
        break
    for dom in domains:
        path = sim_name+"/"+dom[0]
        sim= fepx_sim("sim",path=home+path)
        sim.post_process()
        num_elts = int(sim.sim['**general'].split()[2])
        step = sim.sim['**step']
        #nums= np.arange(1457,1473,1) 373
        #nums= np.arange(261,414,1)

        nums= elts
        #pprint(elts)
        ori = sim.get_output("ori",step="0",res="elts",ids=nums)
        ori10 = sim.get_output("ori",step="10",res="elts",ids=nums)
        ori20 = sim.get_output("ori",step="20",res="elts",ids=nums)
        ori28 = sim.get_output("ori",step="28",res="elts",ids=nums)
        file= open(home+path+"oris0.txt","w")
        file10= open(home+path+"oris10.txt","w")
        file20= open(home+path+"oris20.txt","w")
        file28= open(home+path+"oris28.txt","w")
        for i in ori[0]:
            var=""
            var10=""
            var20=""
            var28=""
            for j in range(3):
                var+= str(ori[0][i][j])+" "
                var10+= str(ori10[0][i][j])+" "
                var20+= str(ori20[0][i][j])+" "
                var28+= str(ori28[0][i][j])+" "
            #print(var)
            file.write(var+"\n")
            file10.write(var10+"\n")
            file20.write(var20+"\n")
            file28.write(var28+"\n")
        file.close()
        file10.close()
        file20.close()
        file28.close()
os.system("pwd")
os.chdir(home)
os.system("./common_files/elt_spred.sh")
exit(0)
for i in [0,25,50]:
    plot_yield_stress(i)
    plot_eff_strain(i)

#
#sim.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
#sim_iso= fepx_sim("name",path=home+"/1_uniaxial")
#sample_num = int(int(val[2])/10)
sample_num= 2000
start = 0
bin_size = 10
max = 0.00004
bins=np.arange(0,max,max/bin_size)
domains = [["Cube","CUB"], ["Elongated","ELONG"]]
#ids = np.arange(start,start+sample_num,1)
home ="/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/"

simulations = os.listdir(home)
simulations.sort()
bulk=False
debug = False

P = [ calculate_schmid(CUB_111[i],CUB_110[i]) for i in range(12)]
#pprint(P)

altered = [0, 1, 2, 3]
unaltered = [4, 5, 6, 7, 8, 9, 10, 11]
def calc_eff_pl_str(sim,domain,under=""):
    file = open(home+sim+"/"+domain+"_eff_pl_str.csv","w")
    sim= fepx_sim(sim,path=home+sim+"/"+domain)
    sim.post_process()
    num_elts = int(sim.sim['**general'].split()[2])
    step = sim.sim['**step']
    slip = sim.get_output("slip",step="28",res="elts",ids="all")
    elt_vol = sim.get_output("elt"+under+"vol",step="0",res="elts",ids="all")
    v_tot = sum(elt_vol[1]["0"])
    del elt_vol
    elt_vol_final = sim.get_output("elt"+under+"vol",step=step,res="elts",ids="all")
    mat_par = sim.material_parameters["g_0"]
    del sim

    sim_iso= fepx_sim("name",path=home+"isotropic/"+domain)
    #sim_iso.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
    sim_iso.post_process()
    baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])
    val =sim_iso.sim["**general"].split()
    baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])
    #stress_iso = [normalize(sim_iso.get_output("stress",step=step,res=res,ids=ids)[str(i)]) for i in ids]
    del sim_iso

    strength = [ float(i.split("d")[0]) for i in mat_par]
    #exit(0)
    altered  =  []
    ratio=1
    for index,val in enumerate(strength):
        if val>baseline:
            altered.append(index)
            ratio = val/baseline
    avg_eff_pl_str_alt = []
    avg_eff_pl_str_unalt = []
    print("***---***")
    print(altered)
    print(baseline)
    print("***---***")
    values = "elt_vol, tot_vol, vol_frac, eff_pl_alt, eff_pl_unalt, vol_eff_pl_alt, vol_eff_pl_unalt"
    file.write(values+"\n")
    for el in range(num_elts):

        total_altered = 0
        total_unaltered = 0

        for i in range(12):
            schmid_val = P[i]
            if i in altered:
                shear_val = slip[0][str(el)][i]
                total_altered+= schmid_val*shear_val
                if debug:
                    print("\n+++Schmid val")
                    pprint(schmid_val, preamble="+++")
                    print("\n\n+++===slip system shear\n===---", shear_val)
                    print("\n+++===Total resolved shear strain")
                    pprint(total_altered, preamble="+++===---")
                    print("-----------------------------------##-------##-------##")
                    print("-----------------------------------##-------##-------##")
                    print("-----------------------------------##-------##\n\n")
            #print("-----------------------------------##-------##-------##")
            #
            else:
                shear_val = slip[0][str(el)][i]
                total_unaltered+= schmid_val*shear_val
                if debug:
                    print("\n+++Schmid val")
                    pprint(schmid_val, preamble="+++")
                    print("\n\n+++===slip system shear\n===---", shear_val)
                    print("\n+++===Total resolved shear strain")
                    pprint(total_unaltered, preamble="+++===---")
                    print("-----------------------------------##-------##-------##")
                    print("-----------------------------------##-------##-------##")
                    print("-----------------------------------##-------##\n\n\n")

        eff_pl_str_alt = math.sqrt((2/3)*inner_prod(total_altered,total_altered))
        eff_pl_str_unalt = math.sqrt((2/3)*inner_prod(total_unaltered,total_unaltered))

        v_el = elt_vol_final[0][str(el)][0]
        v_frac = v_el/v_tot

        avg_eff_pl_str_alt.append(eff_pl_str_alt*v_frac)
        avg_eff_pl_str_unalt.append(eff_pl_str_unalt*v_frac)

        if debug:
            print("el vol", v_el)
            print("tot vol", v_tot)
            print("vol_frac", v_frac)
            print("-----------------------------------##-------##\n")
            print("\n Effective plastic altered :",eff_pl_str_alt)
            print("\n Effective plastic altered :",eff_pl_str_unalt)
            print("-----------------------------------##-------##\n\n")
            print("-----------------------------------##-------##\n")
            print("\n Vol avg Effective plastic altered :",avg_eff_pl_str_alt[el])
            print("\n Vol avg Effective plastic altered :",avg_eff_pl_str_unalt[el])
            print("-----------------------------------##-------##\n\n")

        values = str(v_el)+"," + str(v_tot)+","+ str(v_frac)+","+ str(eff_pl_str_alt)+ ","+ str(eff_pl_str_unalt)+ ","+ str(avg_eff_pl_str_alt[el])+ ","+ str(avg_eff_pl_str_unalt[el])
        file.write(values+"\n")
    print("\n__")
    print(sum(avg_eff_pl_str_alt))
    print(sum(avg_eff_pl_str_unalt))


sim="isotropic"
domain="Cube"
#calc_eff_pl_str(sim,domain,under="_")
domain="Elongated"
#calc_eff_pl_str(sim,domain,under="_")

for i in simulations:
    if i=="common_files":
        print("common_files")
        break
    for dom in domains:
        calc_eff_pl_str(i,dom[0])
#pprint(elt_vol[1]["0"],max=100)


#slip_vs_aniso(i,"Cube", slip_systems,debug=True, save_plot=False,df=dataframe)
exit(0)

file_grain_ids = open(home+"common_files/Cube_grain_elts","w")
elts = []
grain_elts=[]
values = np.arange(0,2000,1)
for val in values:
    for i in file:
        vals= i.split()
        if vals[1] == str(val):
            #print(vals[0])
            elts.append(int(vals[0])-1)
    #print(elts)
    print(val,"----")
    grain_elts.append(elts)
    file_grain_ids.write(str(elts)[1:-1]+"\n")
#print(len(elts))
#print(grain_elts)




dataframe.to_csv("/home/etmengiste/jobs/aps/images/eff_pl_strain_values.csv")
#dataframe= pd.read_csv("/home/etmengiste/jobs/aps/images/eff_pl_strain_values.csv")
ani = [float(i)/100 for i in dataframe["Aniso"]]
for j in range(3):
    fig= plt.figure()
    ax1 = fig.add_subplot(121)
    ax2 = fig.add_subplot(122)
    print(0+25*j,24+25*j)
    plot_eff_strain(j,ax1,"Cube",ani,dataframe)
    plot_eff_strain(j,ax2,"Elongated",ani,dataframe)
    plt.savefig("/home/etmengiste/jobs/aps/images/eff_pl_strain"+slips[j]+"_"+str(i))


exit(0)
#home="/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"
#home="/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/"
#home="/Users/ezramengiste/Documents/neper_fepx_gui/the_sims"
home_full ="/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/"
home ="/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/"
# non first g_0
#home="/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/nonfirst_g_0/slip_study_rerun/"
ist= "/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/isotropic"
























home="/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"
home="/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/"

simulations = os.listdir(home)
simulations.sort()
#sim.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
sim_iso= fepx_sim("name",path=home+"isotropic/Cube")
#sim_iso.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")


sample_num = 2000
start = 0
ids = np.arange(start,start+sample_num,1)

slip_iso =  [normalize(sim_iso.get_output("slip",step="28",res="elsets",ids=ids)[str(i)],absolute=True) for i in ids]
slips_iso = {}

baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])

del sim_iso

fig = plt.figure()
axies = []
for i in range(len(slip_iso[0])):
    slips =[]
    ax= fig.add_subplot(2,6,i+1)
    for id in ids:
        slips.append(slip_iso[id][i])
    slips_iso[str(i)] = slips
    ax.hist(slips,bins=10,color="blue",edgecolor="red",alpha=0.2)
    axies.append(ax)

for sim in ["071", "072", "073", "074","075"]:
    sim= fepx_sim(sim,path=home+sim+"/Cube")

    #sim.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
    slip =  [normalize(sim.get_output("slip",step="28",res="elsets",ids=ids)[str(i)],absolute=True) for i in ids]

    strength = [ float(i.split("d")[0]) for i in sim.material_parameters["g_0"]]
    altered  =  [index for index,val in enumerate(strength) if val>baseline]
    for i in range(len(slip_iso[0])):
        color = "k"
        if i in altered:
            color="red"
        slips =[]
        ax= axies[i]
        ax.clear()
        for id in ids:
            slips.append(slip_iso[id][i])
        slips_iso[str(i)] = slips
        ax.hist(slips,bins=20,color=color ,edgecolor="k", alpha= 0.2)
        ax.set_ylim([0,sample_num*.8])
        ax.set_xlim([0,0.15])

    plt.tight_layout()
    plt.subplots_adjust(left=0.13, right=0.995,top=0.99, bottom=0.1, wspace=0.035)
    plt.savefig("figure"+sim.name)
    del sim

exit(0)

































for i in range(len(slip_iso[0])):
    slips =[]
    ax= fig.add_subplot(2,7,i+1)
    for id in ids:
        slips.append(slip_iso[id][i])
        if i == 0:
            stress_fake_1 =  [ 0.3*float(i) for i in stress_iso[id]]
            plotting_space([stress_fake_1,stress_iso[id]],axis=ax_stress)
    slips_iso[str(i)] = slips
    #print(slips)
    slip_fake_1 = [  abs(float(i-0.002)) for i in slips]
    ax.hist(slip_fake_1,bins=bin_size,color="red",edgecolor="k",alpha=0.2)
    ax.hist(slips,bins=bin_size,color="blue",edgecolor="k",alpha=0.2)
    ax.set_title(slip_systems[i])
    ax.set_ylim([0,int(sample_num)])
    axies.append(ax)


plt.tight_layout()
plt.subplots_adjust(left=0.062, bottom=0.08, right=0.9,top=0.948, wspace=0.162,hspace=.126)
plt.show()
exit(0)
for sim in ["046","050","075"]:
    sim= fepx_sim("name",path=home+sim+"/Cube")
    #sim.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
    slip =  [normalize(sim.get_output("slip",step="28",res="elsets",ids=ids)[str(i)],absolute=True) for i in ids]

    strength = [ float(i.split("d")[0]) for i in sim.material_parameters["g_0"]]
    altered  =  [index for index,val in enumerate(strength) if val>baseline]
    for i in range(len(slip_iso[0])):
        color = "k"
        if i in altered:
            color="red"
        slips =[]
        ax= axies[i]
        for id in ids:
            slips.append(slip_iso[id][i])
        slips_iso[str(i)] = slips
        ax.hist(slips,bins=20,color=color ,edgecolor="k", alpha= 0.2)
        ax.set_ylim([0,sample_num])

plt.tight_layout()
plt.subplots_adjust(left=0.13, right=0.995,top=0.99, bottom=0.1, wspace=0.035)
plt.savefig("figure")

exit(0)


baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])

sims = ["070","071","072","072","074", "075"]
slips_list = {}
for sim in sims:
    slips_list[sim] = slips

elt_num = len(ids)

del sim_iso

for simulation in sims:
    slips = slips_list[simulation]

    sim= fepx_sim("name",path=home+simulation+"/Cube")
    #sim.post_process(options ="neper -S . -reselset slip,crss,stress,sliprate")
    slip =  [normalize(sim.get_output("slip",step="28",res="elsets",ids=ids)[str(i)],absolute=True) for i in ids]

    strength = [ float(i.split("d")[0]) for i in sim.material_parameters["g_0"]]
    altered  =  [index for index,val in enumerate(strength) if val>baseline]
    for j,val in enumerate(slip):
        ax= fig.add_subplot(2,6,j+1)
        slips[str(j)]=val
        color="k"
        if j in altered:
            color="red"
        ax= fig.add_subplot(2,6,j+1)
        ax.hist(slips[str(j)], bins=20 ,color=color,edgecolor=color, alpha=0.5)

        ax.set_ylim([0,1500])
    del sim

#stress = sim.get_output("stress",step="28",res="elsets",ids=ids)
#crss = sim.get_output("crss",step="28",res="elsets",ids=ids)

#pprint(slip,max=100)


#print(len(array_of_ids))
plt.tight_layout()
plt.subplots_adjust(left=0.13, right=0.995,top=0.99, bottom=0.1, wspace=0.035)
plt.show()
plt.savefig("figure")
exit(0)
array_of_ids = []
y = np.arange(6)
y2 = np.arange(12)
for index,i in enumerate(ids):
    if slips[index]> right and slips[index]<=left:
        #print(i,stress[str(i)])
        axs[1].bar(y, stress[str(i)],color="k",edgecolor="k",alpha=0.003)
        axs[2].bar(y2, slip[str(i)],color="k",edgecolor="k",alpha=0.003)
        axs[3].hist(crss[str(i)],color="k",edgecolor="k",alpha=0.003)
        array_of_ids.append(i)


stress = results["stress"]
strain = results["strain"]
