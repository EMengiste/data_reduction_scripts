#--------------------------------------------------------------------------
import sys
sys.path.append("/home/etmengiste/code/data_reduction_scripts/")
from ezmethods import *
import os
import matplotlib.pyplot as plt
import csv
import numpy as np
import pandas as pd

SIZE=65

plt.rcParams.update({'font.size': SIZE})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = 20,15
#
plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  #
denoised= 0
iso_home= "/media/schmid_2tb_1/etmengiste/files/slip_study_rerun/isotropic/Cube"
#iso_home= "/media/schmid_2tb_1/etmengiste/files/update_03_29_2022/current_param"
iso_home ="/media/schmid_2tb_1/etmengiste/files/update_03_29_2022/current_param"
fig, ax = plt.subplots(1, 1)
#
#
offset = 0.002
#
exp= pd.read_csv("fe9cr.csv")

sample_stress=[exp["'stress'"][i]for i in range(0,len(exp["'stress'"]),500)]
sample_strain=[exp["'strain'"][i]for i in range(0,len(exp["'strain'"]),500)]

name= ["current","initial"]
marker= ['-','-']  # Same as '-.'
ymax=0

#stress,strain =get_stress_strain(iso_home,percent=True,strain_rate=1e-2)
sim = fepx_sim("current",path=iso_home)
num= sim.get_num_steps()
stress=[]
strain=[]
for i in range(num):
    stress.append(float(sim.get_output("stress-eq",step=i)[0]))
    strain.append(float(sim.get_output("strain-eq",step=i)[0])*100)
stress[0]=0
ymax= max(stress)
# simulation
ax.plot(strain, stress,"k",linestyle="-",markersize=2,linewidth=4, label="Simiulation")
# Experimental
ax.plot(sample_strain,sample_stress,"ko",markersize=20, label="Experimental")
title = "Optimized Parameters"
stress = '$\sigma_{eq}$'
strain='$'
strain+="\\varepsilon_{eq}"
strain+='$'
x_label = f'{strain} (\%)'
y_label = f'{stress} (MPa)'

# Compile labels for the graphs
plt.ylim([0,201])
plt.xlim([0.00001,2.5])
lines_labels = [a.get_legend_handles_labels() for a in fig.axes]
lines, labels = [sum(lol, []) for lol in zip(*lines_labels)]
#plt.legend(lines, labels,loc="right", fontsize="small")
#plt.title(title,loc='center')
plt.xlabel(x_label)
plt.ylabel(y_label)
#plt.tight_layout()
fig.subplots_adjust(left=0.12, right=0.97,top=0.98, bottom=0.14, wspace=0.1, hspace=0.1)
    
plt.savefig("stress_v_strain.png",dpi=200)
ax.cla()
