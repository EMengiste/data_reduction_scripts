import os
import pandas as pd
from fepx_sim import *
import matplotlib.pyplot as plt
from tool_box import pprint,np

import multiprocessing
import time
# Latex interpretation for plots
# Latex interpretation for plots
plt.rcParams.update({'font.size': 5})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.figsize'] = 5,5

def stress_strain(stresses,strains,show=True):
    fig,axs = plt.subplots(3,3)
    axs = [axs[0][0],axs[1][1],axs[2][2],axs[2][1],axs[2][0],axs[1][0]]
    for i in range(6):
        stress_vals = stresses[:,i]
        strain_vals = strains[:,i]
        axs[i].plot(strain_vals,stress_vals)
    if show==True:
        plt.show()


simulation = fepx_sim("simulation",path="/home/etmengiste/code/Test_updated_code/41_fcc_precipitates.sim")

stress=np.array(simulation.get_output("stress",res="elts",ids="all",step="malory_archer",comp=""))
strain=np.array(simulation.get_output("strain",res="elts",ids="all",step="malory_archer",comp=""))
print(stress.shape)
# step , elt , component
pprint(stress[1,0,:])
stress = np.moveaxis(stress,0,1)
strain = np.moveaxis(strain,0,1)
print(stress.shape)
# step , elt , component
pprint(stress[0,1,:])

yields =[]
for i,j in zip(stress,strain):
    stress_strain(i,j)
    exit(0)