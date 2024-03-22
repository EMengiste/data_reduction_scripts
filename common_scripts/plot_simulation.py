import os
from tkinter import *
from plotting_tools import *
from tool_box import *
from fepx_sim import *
#____Results__availabe___are:
#
# Nodal outputs
#   coo             = [1,2,3]  x,  y,  z
#   disp            = [1,2,3] dx, dy, dz
#   vel             = [1,2,3] vx, vy, vz
# Element outputs (mesh,entity)
#   crss            = [1,2,...,n] where n=12,18,32 for bcc/fcc,hcp,bct respectively
#   rss            = [1,2,...,n] where n=12,18,32 for bcc/fcc,hcp,bct respectively
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
########
#
def plotting_space(x,y=[],layers=3,axis="",resolution=0,ylabel="",debug=False):
    print("-------|---------|---plotting_space--|---------|---------|---------|---------|")
    if y==[]:
        #print(num_components)
        wide=1/(len(x)+2)
        for m in range(len(x)):
            i=x[m]
            y= np.arange(len(i))
            spacing=wide*float(m)
            if axis!="":
                axis.bar(y+spacing,i,width=wide,edgecolor="k")
            if debug:

                print("======\n\n")
                print(m)
                print(y+spacing)
                print(float(x.index(i)))
                print("======\n\n")
                for j in range(len(x[0])):                
                    print(str(j)+"  -------|---------|---------|---------|---------|---------|---------|---------|")
                    for i in range(len(x)):
                        ret= str(i)+" |"
                        print(ret)
                        #print(i)
                        spacer="  |"
                        print(x[i][j])
                        while float(x[i][j])>resolution:
                            ret   +="-.--|"
                            spacer+="---+|"
                            x[i][j]=float(x[i][j])-1
                        for _ in range(layers):
                            print(ret)
            axis.set_ylabel(ylabel)
            #print("-------|---------|---------|---------|---------|---------|---------|---------|")
#
#
def plotter(ax,x,y=""):
    if y=="":
        y=np.arange(len(x))
        ax.bar(y,x)
    else:
        ax.plot(x,y)
    return [x,y]
#
#
def normailze(array,scale=1,maximum=1,absolute=False,debug=False):
    # [f(x) if condition else g(x) for x in sequence]
    if isinstance(array[0],str):
        array=[float(i) 
            if "d" not in i 
            else float(i.split("d")[0])
            for i in array ]        
    if absolute:
        array= [np.abs(i) for i in array]
    if isinstance(array[0],list):
        return array
    
    if debug:
        print("-------|---------|---------|---------|---------|---------|---------|---------|")
        print("-------|---------|---------|---------|---------|---------|---------|---------|")
        print("-------|---------|---------|---------|---------|---------|---------|---------|")

        print(array)
        print("maximum val "+str(maximum)+"------")        
        print("-------|---------|---------|---------|---------|---------|---------|---------|")
        print("-------|---------|---------|---------|---------|---------|---------|---------|")

    if maximum=="":
        maximum = max(array)
    return [scale*i/maximum for i in array]
#
import sys
SIZE=12

plt.rcParams.update({'font.size': SIZE})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Sans'
# plt.rcParams['figure.figsize'] = 6,6
#
plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  #

# comand line inputs
processing= sys.argv[2] == "1"
#
sim_path = sys.argv[1]#"/home/etmengiste/jobs/UAH_collaboration/linear_gradient/001/simulation.sim"
sim=fepx_sim("test_sim",path=sim_path) # object created
sim_results = sim.get_results(process=processing,)
print(sim_results.keys())
try:
    sims= [sys.argv[3]]
    num_results=1
except:
    sims=list(sim_results.keys())[2:]
    num_results=len(sim_results.keys())-2
print("num of results is", num_results)
width = 1
length = math.ceil(num_results/width)
print(length)
# exit(0)
plotted= 1
fig = plt.figure(figsize=[15,20])
length+=1
ax = fig.add_subplot(length,width,plotted)
try:
    stress=sim.get_output("stress",step="all",comp=2)
    strain=sim.get_output("strain",step="all",comp=2)
except:
    sim.post_process(options="-resmesh stress,strain")
    stress=sim.get_output("stress",step="all",comp=2)
    strain=sim.get_output("strain",step="all",comp=2)
plot_stress_strain(ax,stress,strain,lw=1,ls="-.")
plotted+=1
ax.set_xlim([-0.01,0.31])
ax.set_ylim([-0.01,1500])
del sim
for value in sims:        
    steps_val= sim_results[value]
    steps_val = normailze(steps_val,maximum=1,absolute=True,debug=True)
    #
    # exit(0)
    ax = fig.add_subplot(length,width,plotted)
    if isinstance(steps_val,float):
        plotting_space([steps_val],axis=ax,ylabel=value,layers=2)
    elif isinstance(steps_val,list):
        plotting_space(steps_val,axis=ax,ylabel=value,layers=2)       
    print("===+",plotted,"\n")
    ax.set_ylim([0,0.0125])
    plotted+=1
plt.tight_layout()
print(os.getcwd())
fig.savefig(sim_path+"/image",dpi=300)
