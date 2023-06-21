import os
from ezmethods import *
import time
import pandas as pd

unirr_insitu_ff = "/home/etmengiste/jobs/aps/2023/Fe9Cr HEDM data repository/Fe9Cr-61116 (unirr)/in-situ ff-HEDM/"
irr_insitu_ff1 = "/home/etmengiste/jobs/aps/2023/Fe9Cr HEDM data repository/KGT1119 (450C, 0.1 dpa)/In-situ ff HEDM/"
irr_insitu_ff2 = "/home/etmengiste/jobs/aps/2023/Fe9Cr HEDM data repository/KGT1147 (300C, 0.1 dpa)/In-situ ff HEDM/"

preamble= "In-situ FF Parent Grain Data "

#
#
# Generates the ori files in 
for dirs in [unirr_insitu_ff, irr_insitu_ff1, irr_insitu_ff2]:
    os.chdir(dirs)
    dir = [i for i in os.listdir(dirs) if i.endswith(".csv") and i.startswith(preamble)]
    dir.sort()
    # initial
    file = open(dirs+"ini","w")
    csv = pd.read_csv(dirs+dir[0])
    length = len(csv)
    for i in range(length):
        #print(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i]))
        file.write(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i])+"\n")
    # final
    file = open(dirs+"fin","w")
    csv = pd.read_csv(dirs+dir[-1])
    length = len(csv)
    for i in range(length):
        #print(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i]))
        file.write(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i])+"\n")
    # all
    file = open(dirs+"all","w")
    for i in dir:
            print(dirs+i[:-4].replace(" ","_")+"_ori")
            csv = pd.read_csv(dirs+i)
            length = len(csv)
            for i in range(length):
                #print(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i]))
                file.write(str(csv["ROD1"][i])+"  "+str(csv["ROD2"][i])+"  "+str(csv["ROD3"][i])+"\n")
    os.system("~/code/data_reduction_scripts/plot_ipf.sh")
exit(0)
for dirs in [unirr_insitu_ff, irr_insitu_ff1, irr_insitu_ff2]:
    dir = os.listdir(dirs)
    dir.sort()
    dir = [i for i in dir if i.endswith(".csv") and i.startswith(preamble)]
    combined_ipf(dir)
#
exit(0)