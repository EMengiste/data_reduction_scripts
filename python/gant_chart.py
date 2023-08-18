import pandas as pd
import matplotlib.pyplot as plt 

SIZE=25
# Latex interpretation for plots
plt.rcParams.update({'font.size': SIZE})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 400
plt.rcParams['figure.figsize'] = 16,4
plt.rcParams['axes.axisbelow'] = True
#
plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  #

fig, ax = plt.subplots(1, 1,sharey="row")
# name: [width,left]
tasks_time = {"Simulations, verification \nand validation":[1,3],
              "Model testing \nand iteration":[1,2],
              "Model formulation\n and implementation":[2,0.25],
              "Experimental \ncharacterization":[1.25,0]}
for i in tasks_time:
    task=tasks_time[i]
    ax.barh(y=i,width=task[0],left=task[1],color="grey",edgecolor="k")

#xticks=["Fall 2021","Spring 2022","Fall 2022","Spring 2023","Fall 2023","Spring 2024","Fall 2024","Spring 2025","Fall 2025","Spring 2026"]
ax.set_xlim([0,4])
#ax.set_xticklabels(xticks)
fig.subplots_adjust(left=0.18, right=0.97,top=0.9, bottom=0.13, wspace=0.07, hspace=0.1)
ax.grid(axis="x",zorder=9)
plt.savefig("gant_chart.png")