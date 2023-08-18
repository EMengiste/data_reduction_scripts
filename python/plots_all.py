import os
from ezmethods import *
import pandas as pd
import matplotlib.pyplot as plt
import os
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


simulations = os.listdir(home)
simulations.sort()
simulations.remove("common_files")
sim_iso = fepx_sim("Cube.sim",path=home+"isotropic/Cube.sim")
step= "27"
num_steps = sim_iso.get_num_steps()
destination = "/home/etmengiste/jobs/aps/eff_pl_strain/"
domain = "Cube.sim"
calc=True
marker_size=40
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
        print(i)
        name=aps_home+"sim_"+domain+"_"+str(i)+"_eff_pl_str"
        print(name)
        ratios= [1]
        altered= [1]
        unaltered= [1]
        for next in range(1,7):
            sim = "00"+str(i+next)
            if calc:
                alt,unalt,ratio = calc_eff_pl_str(sim[-3:],domain,out=True,step=step,home=home,iso_home=home, debug=False)
            else:
                alt = pd.read_csv(home+sim+"/"+domain+"_alt_eff_pl_str.csv",sep=",")
                #iso = pd.read_csv(home+sim+"/"+domain+"_alt_eff_pl_str.csv",sep=",")
                print(alt)
                print(iso)
                exit(0)
            exit(0)
            ratios.append(ratio)
            altered.append(alt)
            unaltered.append(unalt)
        print(ratios)
        print(altered)
        print(unaltered)
        name = "sim_"+domain+"_"+str(i)+"_eff_pl_str"
        print("--")
        print("altered ",altered)
        print("unaltered ",unaltered)
        set =int((i-30*j)/6)
        print(set)
        ax.plot(ratios,unaltered,"k-o",ls=sets[set],ms=marker_size,label="Set "+str(set+1))
        ax.plot(ratios,altered,"kD",ls=sets[set],ms=marker_size)
        ax.set_xticks(ratios)
        ax.set_xticklabels(an,rotation=90)     
        ax.set_yticks([0,0.5,1,1.5,2])
        ax.set_yticklabels(["0.00","0.50","1.00","1.50","2.00"])  

y_label="$\\bar\\varepsilon^{p}$ (-)"
axs[0].set_ylabel(y_label,labelpad=25)
#ax.legend()


fig.supxlabel(x_label,fontsize=SIZE)
fig.subplots_adjust(left=0.09, right=0.98,top=0.9, bottom=0.2, wspace=0.07, hspace=0.1)

plt.savefig("eff_pl_strain_"+domain)
exit(0)

plot_eff_strain(0,all=True,marker_size=15)
#plot_yield_stress(0,all=True,marker_size=15)

exit(0)

