
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
domains = [["Cube","CUB"]]
#
aniso = ["125","150","175","200","400"]

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
home ="/media/schmid_1tb_2/etmengiste/aps_add_slip/"
simulations = os.listdir(home)
simulations.sort()
simulations =simulations[:-3]
bulk=True
length= len(simulations)
print(simulations[-10:])
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
#   defrate_pl      = [1,2,3,4,5,6] tensor
#   defrate_pl_eq   = [1]
#   elt_vol         = [1]
#   ori             = [1,..,n] where n= 3 if rod of euler or 4 if quat or axis angle
#   slip            = [1,2,...,n] where n=12,18,32 for bcc/fcc,hcp,bct respectively
#   sliprate        = [1,2,...,n] where n=12,18,32 for bcc/fcc,hcp,bct respectively
#   spinrate        = [1,2,3] skew symetric plastic spin rate tensor
#   strain          = [1,2,3,4,5,6] tensor
#   strain_eq       = [1]
#   strain_el       = [1,2,3,4,5,6] tensor
#   strain_el_eq    = [1]
#   strain_pl       = [1,2,3,4,5,6] tensor
#   strain_pl_eq    = [1]
#   stress          = [1,2,3,4,5,6] tensor
#   stress_eq       = [1]
#   velgrad         = [1,2,3,4,5,6,7,8,9] full velocity gradient tensor
#   work            = [1]
#   work_pl         = [1]
#   workrate        = [1]
#   workrate_pl     = [1]
########
########
########

def post_process(domains,length):
    # for saving  not useful (guess i lied it's useful for slip vs aniso)
    #aniso = {"Aniso":["100"]+aniso}
    #dataframe = pd.DataFrame(aniso)
    #data = pd.DataFrame(aniso)

    #for i in range(5,10,5):
    for i in range(0,length,5):
        if bulk:
            for aniso_index in range(5):
                current= i+aniso_index
                simulation = simulations[current]
                print("*****-----+++++",simulation)
                num_slips  = "4"#slips[int(current/25)]
                if current>=20:
                    num_slips = "6"
                    set_num    = str(int((current/5)+1)-4)
                else:
                    set_num    = str(int((current/5)+1))
                for domain in domains:
                    # DOM_CUB_NSLIP_2_SET_1_ANISO_125
                    name = "DOM_"+domain[1]+"_NSLIP_"+num_slips+"_SET_"+set_num+"_ANISO_"+aniso[aniso_index]

                    calc_eff_pl_str(simulation,"Cube.sim",home=home)
                    #get_bulk_output(simulation,domain[0])
                    #package_oris(home+simulation+"/"+domain[0]+".sim/", name=name)

        print(i,"====\n\n\n")
        print("get_yield_v_aniso_ratio(i,\"Cube\")")
        #get_yield_v_aniso_ratio(i,"Cube",home=home,plot=True)
        #
        #get_yield_v_aniso_ratio(i,"Elongated")

        print("calc_eff_pl_str(simulations[i],\"Cube\")")
        #calc_eff_pl_str(simulations[i],"Cube.sim",home=home)
        #
        #calc_eff_pl_str(simulations[i], "Elongated")

        # not good dont use requires old data structure
        print("slip_vs_aniso(i,\"Cube\", slip_systems,debug=\"comparision\", save_plot=True,df=dataframe)")
        #slip_vs_aniso(i,"Cube.sim", slip_systems,debug="comparision", save_plot=True)
        #
        #slip_vs_aniso(i,"Elongated", slip_systems,debug="comparision", save_plot=True, df=dataframe)
    #plt.savefig("/home/etmengiste/jobs/aps/images/Combined_plot"+simulations[i]+"_")

####
post_process(domains,145)
#
