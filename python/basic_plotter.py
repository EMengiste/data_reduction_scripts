import matplotlib.pyplot as plt
import os
import math
plt.rcParams.update({'font.size': 160})
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.dpi"] = 220
plt.rcParams["figure.subplot.left"] = 0.045
plt.rcParams["figure.subplot.bottom"] = 0.08
plt.rcParams["figure.subplot.right"] = 0.995
plt.rcParams["figure.subplot.top"] = 0.891
plt.rcParams["figure.subplot.wspace"] = 0.21
plt.rcParams["figure.subplot.hspace"] = 0.44
plt.rcParams['figure.figsize'] = 45,30
import pandas as pd
import numpy as np

home ="/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/"

simulations = os.listdir(home)
simulations.sort()

for simulation in simulations[:1]:
    for dom in ["Cube", "Elongated"]:
        if simulation!= "common_files":
            file=home+simulation+"/"+dom+"_eff_pl_str.csv"
            print("opening file ",file)
            data = pd.read_csv(file)
            tot_alt = sum(data[" vol_eff_pl_alt"])
            tot_unalt = sum(data[" vol_eff_pl_unalt"])
            print("--[+ total altered = ",tot_alt)
            print("--[+ total unaltered = ",tot_unalt)
exit(0)
x_lim= 2
x_label="r"
y_label="$\\tau_{crss}$"
x= np.arange(0,x_lim,0.0001)
tau_shear = x**0.5
tau_cut = 1/x

plt.plot(x,tau_cut,"b",lw=5)
plt.plot(x,tau_shear,"r",lw=5)
#plt.hlines(0.3,xmin=0,xmax=x_lim)
plt.xlim([-0.0001,x_lim])
plt.ylim([0,1])
plt.xticks([])
plt.yticks([])
plt.xlabel(x_label)
plt.ylabel(y_label)
#plt.show()
plt.savefig("/home/etmengiste/jobs/cut_bow")