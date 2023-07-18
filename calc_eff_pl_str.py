import os
from ezmethods import *
import pandas as pd
import matplotlib.pyplot as plt
import os

import multiprocessing
import time
SIZE=35
# Latex interpretation for plots
plt.rcParams.update({'font.size': SIZE})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 400
plt.rcParams['figure.figsize'] = 23,8
#
plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  #
# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1
    

iso_home="/home/etmengiste/jobs/aps/slip_study/"
home="/media/schmid_2tb_1/etmengiste/files/slip_system_study/"

#

## Plot effective plastic strain
def plot_eff_strain(start,step,all=False,marker_size=15,aps_home="/home/etmengiste/jobs/aps/eff_pl_str/"):
    sets    =  ["solid","dotted","dashdot",(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))]
    an = ["Iso.", "1.25", "1.50", "1.75", "2.00", "3.00", "4.00"]
    for domain in ["Cube"]:
        if all:
            fig, axs = plt.subplots(1, 3,sharey="row")
            for j in range(3):
                ax = axs[j]
                ax.cla()
                ax.set_ylim([-0.1,2.1])
                ax.set_xlim([0.9,4.1])
                slip = slips[j]
                ax.set_title(slip+" slip systems strengthened")
                print(j+1)       
                for i in range(30*j,30*j+30,6):
                    #print(dat)
                    print("--")
                    vals = start[i:i+6]
                    ratios =  [1]+list(vals["ratio"])
                    altered=  [1]+list(vals["altered"])
                    unaltered = [1]+list(vals["unaltered"])
                    print("altered ",altered)
                    print("unaltered ",unaltered)
                    set =int((i-30*j)/6)
                    print(set)
                    print(sets[set])
                    ax.plot(ratios,unaltered,"ko",linestyle=sets[set],ms=marker_size,label="Set "+str(set+1))
                    ax.plot(ratios,altered,"kD",linestyle=sets[set],ms=marker_size)
                    ax.set_xticks(ratios)
                    ax.set_xticklabels(an,rotation=90)     
                    ax.set_yticks([0,0.5,1,1.5,2])
                    ax.set_yticklabels(["0.00","0.50","1.00","1.50","2.00"])  

            y_label="$\\bar\\varepsilon^{p}$ (-)"
            axs[0].set_ylabel(y_label,labelpad=25)
        else:
            fig= plt.figure()
            ax = fig.add_subplot(111)
            ax.cla()
            ax.set_ylim([0,2.1])
            ax.set_xlim([0.9,4.1])
            for i in range(start,start+25,5):
                dat = pd.read_csv(aps_home+"sim_"+domain+"_"+str(i)+"_eff_pl_str",index_col=0)
                #print(dat)
                print("--")
                ratios= dat.iloc[0]
                altered= dat.iloc[1]
                unaltered= dat.iloc[2]
                print(altered)
                print(unaltered)
                set =int((i-start)/5)
                ax.plot(ratios,unaltered,"k-*",ls=sets[set],ms=25,lw=4,label="Set "+str(set+1))
                ax.plot(ratios,altered,"D",ls=sets[set],ms=25,lw=4)
                ax.set_title(domain)
                ax.set_xticks(ratios)
                ax.set_xticklabels(an)

            sli = slips[int(start/25)]
        #ax.legend()
        y_label="$\\bar\\varepsilon^{p}$ (-)"


        fig.supxlabel(x_label,fontsize=SIZE)
        fig.subplots_adjust(left=0.09, right=0.98,top=0.9, bottom=0.2, wspace=0.07, hspace=0.1)
    
        plt.savefig(aps_home+"eff_pl_strain_"+domain+step)
    #plt.show()
  #
 #
#


def calc_eff_pl_str(params,step="28",out=True,
                    name="",under="",home=home,
                    iso_home=home, debug=False):
    sim,domain,step =params
    sim_val = sim
    if name=="":
        name= domain[:-4]
    file = open(home+sim+"/"+name+"_alt_eff_pl_str.csv","w")

    file_iso = open(home+sim+"/"+name+"isotropic"+"_eff_pl_str.csv","w")
    #
    sim= fepx_sim(sim,path=home+sim+"/"+domain)
    sim.post_process()
    num_elts = int(sim.sim['general'].split()[2])
    slip = sim.get_output("slip",step=step,res="elts",ids="all")
    elt_vol = sim.get_output("elt"+under+"vol",step="0",res="elts",ids="all")
    v_tot = sum([i[0] for i in elt_vol])
    elt_vol_final = sim.get_output("elt"+under+"vol",step=step,res="elts",ids="all")
    mat_par = sim.material_parameters["g_0"]
    del sim
    #   ISO
    sim_iso= fepx_sim("name",path=iso_home+"isotropic/"+domain)
    sim_iso.post_process()
    slip_iso = sim_iso.get_output("slip",step=step,res="elts",ids="all")
    elt_vol_iso = sim_iso.get_output("eltvol",step="0",res="elts",ids="all")
    v_tot_iso = sum([i[0] for i in elt_vol_iso])
    elt_vol_final_iso = sim_iso.get_output("eltvol",step=step,res="elts",ids="all")
    baseline = float(sim_iso.material_parameters["g_0"][0].split("d")[0])
    #stress_iso = [normalize(sim_iso.get_output("stress",step=step,res=res,ids=ids)[str(i)]) for i in ids]
    del sim_iso
    #
    strength = [ float(i.split("d")[0]) for i in mat_par]
    #exit(0)
    altered  =  []
    ratio=1
    for index,val in enumerate(strength):
        if val>baseline:
            altered.append(index)
            ratio = val/baseline
    print(ratio)
    #
    #
    avg_eff_pl_str_alt = []
    avg_eff_pl_str_unalt = []
    #
    avg_eff_pl_str_alt_iso = []
    avg_eff_pl_str_unalt_iso = []
    #
    print("***---***")
    #print(altered)
    #print(baseline)
    print("***---***")
    values = "elt_vol, tot_vol, vol_frac, eff_pl_alt, eff_pl_unalt, vol_eff_pl_alt, vol_eff_pl_unalt"
    file.write(values+"\n")
    file_iso.write(values+"\n")
    #
    for el in range(num_elts):
        total_altered = 0
        total_unaltered = 0
        total_altered_iso = 0
        total_unaltered_iso = 0
        #
        for i in range(12):
            schmid_val = P[i]
            shear_val = slip[el][i]
            shear_val_iso = slip_iso[el][i]
            if i in altered:
                total_altered+= schmid_val*shear_val
                total_altered_iso+= schmid_val*shear_val_iso
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
                total_unaltered+= schmid_val*shear_val
                total_unaltered_iso+= schmid_val*shear_val_iso
                if debug:
                    print("\n+++Schmid val")
                    pprint(schmid_val, preamble="+++")
                    print("\n\n+++===slip system shear\n===---", shear_val)
                    print("\n+++===Total resolved shear strain")
                    pprint(total_unaltered, preamble="+++===---")
                    print("-----------------------------------##-------##-------##")
                    print("-----------------------------------##-------##-------##")
                    print("-----------------------------------##-------##\n\n\n")
        #
        eff_pl_str_alt = math.sqrt((2/3)*inner_prod(total_altered,total_altered))
        eff_pl_str_unalt = math.sqrt((2/3)*inner_prod(total_unaltered,total_unaltered))
        #
        eff_pl_str_alt_iso = math.sqrt((2/3)*inner_prod(total_altered_iso,total_altered_iso))
        eff_pl_str_unalt_iso = math.sqrt((2/3)*inner_prod(total_unaltered_iso,total_unaltered_iso))
        #
        #
        v_el = elt_vol_final[el][0]
        v_frac = v_el/v_tot
        #
        #iso
        v_el_iso = elt_vol_final_iso[el][0]
        v_frac_iso = v_el_iso/v_tot_iso
        #
        # altered
        avg_eff_pl_str_alt.append(eff_pl_str_alt*v_frac)
        avg_eff_pl_str_unalt.append(eff_pl_str_unalt*v_frac)
        #
        #iso
        avg_eff_pl_str_alt_iso.append(eff_pl_str_alt_iso*v_frac_iso)
        avg_eff_pl_str_unalt_iso.append(eff_pl_str_unalt_iso*v_frac_iso)
        #
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
        #
        values = str(v_el)+"," + str(v_tot)+","+ str(v_frac)+","+ str(eff_pl_str_alt)+ ","+ str(eff_pl_str_unalt)+ ","+ str(avg_eff_pl_str_alt[el])+ ","+ str(avg_eff_pl_str_unalt[el])
        file.write(values+"\n")
        #
        values = str(v_el_iso)+"," + str(v_tot_iso)+","+ str(v_frac_iso)+","+ str(eff_pl_str_alt_iso)+ ","+ str(eff_pl_str_unalt_iso)+ ","+ str(avg_eff_pl_str_alt_iso[el])+ ","+ str(avg_eff_pl_str_unalt_iso[el])
        #print(values)
        file_iso.write(values+"\n")
    file.close()
    file_iso.close()
    print("\n__")
    file.close()
    file_iso.close()
    if out:            
        return [sim_val,sum(avg_eff_pl_str_alt)/sum(avg_eff_pl_str_alt_iso),sum(avg_eff_pl_str_unalt)/sum(avg_eff_pl_str_unalt_iso), ratio]
  #
 #
#
##
debug=False
iso_home="/home/etmengiste/jobs/aps/slip_study/"
home="/media/schmid_2tb_1/etmengiste/files/slip_system_study/"

simulations = os.listdir(home)
simulations.sort()
simulations.remove("common_files")
step= "27"

destination = "/home/etmengiste/jobs/aps/eff_pl_strain/"
pool = multiprocessing.Pool(processes=90)


means = []

stds = []
set_start = 0
set_end = 5
headers = ["sim_name","altered","unaltered","ratio"]
data = [headers]
###
aniso = ["125", "150", "175", "200", "300", "400"]
slips = ["2","4","6"]
sets = ["1","2","3","4","5"]
num_sets = len(sets)
base = 6
domain = "Cube.sim"
generate = False
#generate = True
base = len(simulations[:90])
set_of_sims = [simulations[m] for m in range(0,base,1)]
sims = np.array([set_of_sims,np.tile(domain,(base)),np.tile(step,(base))]).T

if debug:
    num_5=calc_eff_pl_str(["005","Cube.sim","27"])
    num_3=calc_eff_pl_str(["003","Cube.sim","27"])
    print(num_3,num_5)
    exit(0)


print("starting code")
tic = time.perf_counter()
#
name = "eff_pl_str"+step
print(tic)
if generate:
    value = pool.map(calc_eff_pl_str,sims)
    data +=value   
    df = pd.DataFrame(data)
    df.columns=df.iloc[0]
    df[1:].to_csv(destination+name+".csv")
else:
    df = pd.read_csv(destination+name+".csv")
    plot_eff_strain(df,step,aps_home=destination,all=True)
toc = time.perf_counter()
print("===")
print("===")
print("===")
print(f"Generated data in {toc - tic:0.4f} seconds")
#exit(0)
print("===")
print("===")
print("===")
print("starting plotting")
tic = time.perf_counter()
### misori plot code

exit(0)