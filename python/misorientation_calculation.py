import os
from ezmethods import *
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt

import multiprocessing
import time
# Latex interpretation for plots
# Latex interpretation for plots
plt.rcParams.update({'font.size': 55})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.figsize'] = 20,20

# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1


remote= "/run/user/1001/gvfs/sftp:host=acmelabpc2.eng.ua.edu,user=etmengiste"
remote = ""

iso_home = remote+"/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"

home=remote+"/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_system_study/"


def plot_mean_data(NAME,ylims="",y_label="",
                       unit="",y_ticks ="",y_tick_lables="",debug=False):
    norm =False
    result ={}
    results = pd.read_csv(NAME+".csv").transpose()
    for i in results:
        result[results[i]["case"]]= [float(results[i]["mean"]),float(results[i]["std"])]
        #print(results[i]["case"],result[results[i]["case"]])
    DOMAIN = [ "CUB"]
    DOM = [ "cubic"]
    NSLIP  = ["2","4", "6"]
    #NSLIP =["2"]
    ANISO  = ["125", "150", "175", "200", "400"]
    SETS    = ["1", "2", "3", "4", "5"]
    an = ["1.25", "1.50", "1.75", "2.00", "3.00", "4.00"]
    #SETS = ["1"]
    sets    =  ["solid","dotted","dashdot",(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))]
    #
    ###
    aniso = [125, 150, 175, 200, 300, 400]
    differences= {}
    difs2={}

    if debug:
        NSLIP =["2"]
        DOMAIN = ["CUB"]
        DOM = ["cubic"]

    for dom in DOMAIN:
        fig, axs = plt.subplots(1, 3,sharex="col",sharey="row",figsize=( 23,8))
        for slip in NSLIP:
            #
            ax0= axs[NSLIP.index(slip)]
            #ax1= axs[1][NSLIP.index(slip)]
            #
            ax0.set_title(slip+" slip systems strengthened")
            #
            ax0.set_xlim([115,410])
            #ax1.set_xlim([90,410])
            if ylims!="":
                ax0.set_ylim(ylims)
                #ax1.set_ylim([0.95,2.0])
            else:
                ax0.set_ylim([4.6,15.1])
                #ax1.set_ylim([-0.4,10.4])
            #fig, ax = plt.subplots(1, 1,figsize=(10,12))
            for set,line in zip(SETS,sets):
                #
                index1 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_125"]
                index2 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_150"]
                index3 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_175"]
                index4 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_200"]
                index5 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_300"]
                index6 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_400"]

                list1 = [index1[0], index2[0],
                    index3[0], index4[0], index5[0],index6[0]]
                list2 =[index1[1], index2[1],
                    index3[1], index4[1], index5[1],index6[1]]
                
                differences[dom+"_"+slip+"_"+set] = list1
                difs2[dom+"_"+slip+"_"+set] = list2
                #print(list)
                #
                if norm == True:
                    list1 = [i/result["DOM_"+dom+"_ISO"][0] for i in list1]
                    list2 = [i/result["DOM_"+dom+"_ISO"][1] for i in list2]
                #### Start of plotting code
                #
                ax0.plot(aniso,list1,"k",lw=2,linestyle=line,
                    label="Set "+str(set))
                #ax1.plot(aniso,list2,"k",lw=2,linestyle=line,
                #    label="Set "+str(set))
                #
                marker_size =130
                #
                for a, l, l2 in zip(aniso, list1, list2):
                    ax0.scatter(a, l, marker="o", s=marker_size,edgecolor="k",color= "k")
                    #ax1.scatter(a, l2, marker="o", s=marker_size,edgecolor="k",color= "k")
                #
                ax0.set_xticks(aniso)
                ax0.set_xticklabels(an,rotation=90)

                #ax1.set_xticks(aniso)
                #ax1.set_xticklabels(an,rotation=90)
        if norm and unit!="":
            unit = "(-)"
        axs[0].set_ylabel(y_label,labelpad=25)

        if y_tick_lables !="":
            axs[0].set_yticks(y_ticks)
            axs[0].set_yticklabels(y_tick_lables)
            axs[0].set_ylabel(y_label+unit,labelpad=25)
        ###
        #
        # Put title here
        #title = "Deviation from Taylor Hypothesis: \n"+str(DOM[DOMAIN.index(dom)])+" domain, "+str(slip)+" slip systems\n"
        #
        deg = "$^{\circ}$"
        #
        x_label = f'$p$ (-)'
        #y_label = f'Normalized Misorienation ({deg})'
        fig.supxlabel(x_label,fontsize=SIZE)
        fig.subplots_adjust(left=0.09, right=0.98,top=0.9, bottom=0.2, wspace=0.07, hspace=0.1)
        fig.savefig(NAME+str(DOM[DOMAIN.index(dom)])+"_mean.png",dpi=400)
    


home=remote+"/media/schmid_2tb_1/etmengiste/files/slip_system_study/"


#name = "calculation_stress_misori"
#ax = plt.figure().add_subplot(projection='3d')
#plot_std_mean_data(name,debug=False)
#exit(0)

def calc_misorientation_angles(params):
       #print(params)
       sim_ind,base =params
       sim_name = simulations[sim_ind]
       sim = fepx_sim("Cube.sim",path=home+sim_name+"/Cube.sim")

       set = str(sets[int(sim_ind/base)%num_sets])
       #print(sim_name,sim_ind)
       slip = str(slips[int(sim_ind/(base*num_sets))])  

       ori_iso = sim_iso.get_output("ori",step=step,res="elsets",ids="all")
       #print(stress_iso)
       ori_iso = R.from_mrp(np.array(ori_iso)).as_matrix()

       ori = sim.get_output("ori",step=step,res="elsets",ids="all")
       ori = R.from_mrp(np.array(ori)).as_matrix()
       length = len(ori)
       print(length)
       misori = []
       for index in range(length):
            o_iso,o = ori_iso[index],ori[index]
            fill = '█'
            percent = round(index / float(length-1),3)
            percent*=100
            prefix=" \n\n==working<"
            misori.append(dif_degs(o_iso,o))
            print('\r{[%d%%]\r'%index, end="")
       name = "DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_"+aniso[sim_ind%6]
       print(name)
       vals = [name ,np.mean(misori),np.std(misori)]
       del sim
       return vals


simulations = os.listdir(home)
simulations.sort()
simulations.remove("common_files")
sim_iso = fepx_sim("Cube.sim",path=home+"isotropic/Cube.sim")
steps = [str(i) for i in range(0,28)]

res = [0 for i in range(90)]
res2 = [0 for i in range(90)]
new_res = {}

for step in steps:
    print(step)

    NAME = "calculation_misori_"+step
    result ={}
    results = pd.read_csv(NAME+".csv").transpose()
    for i in results:
        result[results[i]["case"]]= [float(results[i]["mean"]),float(results[i]["std"])]
        res[i]= res[i]+result[results[i]["case"]][0]
        new_res[results[i]["case"]]=[res[i]+float(results[i]["mean"]),res2[i]+float(results[i]["std"])]
        print(i)

###
headers = ["case","mean","std"]
data = [headers]
###
for i in new_res:
    data.append([i,new_res[i][0]/28,new_res[i][1]/28])

#
name= "the_file"
pprint(data)
df = pd.DataFrame(data)
df.columns=df.iloc[0]
df[1:].to_csv("the_file.csv")


y_lab ="$\\theta$ (MPa)"
ylims = [0,3]

y_ticks = [5.00,7.50,10.00,12.50,15.00]
y_tick_lables = ["5.00","7.50","10.00","12.50","15.00"]


plot_mean_data(name,y_label=y_lab,ylims=ylims,debug=False)
exit(0)

for step in steps:
    means = []
    stds = []
    set_start = 0
    set_end = 5
    headers = ["case","mean","std"]
    data = [headers]
    ###
    aniso = ["125", "150", "175", "200", "300", "400"]
    slips = ["2","4","6"]
    sets = ["1","2","3","4","5"]
    num_sets = len(sets)
    base = 6
    dom = "CUB"

    pool = multiprocessing.Pool(processes=90)
    print("starting code")
    tic = time.perf_counter()
    #
    #   main_code
    num__base=6
    base = len(simulations[:90])
    set_of_sims = [m for m in range(0,base,1)]
    sims = np.array([set_of_sims,np.tile(num__base,(base))]).T
    value = pool.map(calc_misorientation_angles,sims)
    #value = calc_misorientation_angles(sims[-1])
    data +=value   

    toc = time.perf_counter()
    print(f"Ran the code in {toc - tic:0.4f} seconds")
    print("starting plotting")
    tic = time.perf_counter()
    #stress_calculation code
    name = "calculation_misori_"+step
    y_lab ="$\\theta$ (MPa)"
    ylims = [0,3]

    y_ticks = [5.00,7.50,10.00,12.50,15.00]
    y_tick_lables = ["5.00","7.50","10.00","12.50","15.00"]


    df1 = pd.DataFrame(data)
    df1.columns=df1.iloc[0]
    df1[1:].to_csv(name+".csv")

    #ax = plt.figure().add_subplot(projection='3d')
    plot_mean_data(name,y_label=y_lab,ylims=ylims,debug=False)


    toc = time.perf_counter()
    print(f"Generated plot in {toc - tic:0.4f} seconds")
    print("end of code!")

del sim_iso
exit(0)

##############

fig, ax = plt.subplots(1, 2,sharey="row")

ax[0].plot([1.25,1.5,1.75,2,4],means)

ax[1].plot([1.25,1.5,1.75,2,4],stds)
plt.show()
exit(0)

start = np.array([[-0.92396065,  0.34995201,  0.03178429],
         [0.37975053, 0.88091702, 0.04159496],
         [-0.04567549, -0.31862015,  0.99862887]]).T


fin =np.array([ [-0.9938902,  -0.10962086,  0.01487836],
              [ 0.00147866,  0.21194585, -0.9997406 ],
              [-0.1103634,   0.97111391, -0.01724456]]).T

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
              dat = pd.read_csv(name,index_col=0)
              #print(dat)
              print("--")
              ratios= dat.iloc[0]
              altered= dat.iloc[1]
              unaltered= dat.iloc[2]
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


remote= "/run/user/1001/gvfs/sftp:host=acmelabpc2.eng.ua.edu,user=etmengiste"
home=remote+"/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"


simulations = os.listdir(home)
simulations.sort()
simulations.remove("common_files")
colors = ["k", (10/255,10/255,10/255),
          (70/255,70/255,70/255),
          (150/255,150/255,150/255),
          (200/255,200/255,200/255),
          (230/255,230/255,230/255)]
#


#start = np.array([[1,0,0],[0,1,0],[0,0,1]])


value = np.dot(start,np.linalg.inv(start))
print(value)
print(np.linalg.tensorsolve(start,fin))
exit(0)
Q_ij, Q_ji =rot_mat(start,fin)
pprint(Q_ij)
pprint(Q_ji)
pprint(start,preamble="start")
pprint(fin,preamble="end")
pprint(Q_ij,preamble="q_ij")
pprint(Q_ji,preamble="q_ji")
print( np.dot(Q_ij,start[0]))
print( np.dot(Q_ji,fin[0]))

val1 = np.dot(Q_ij,start[0])
ax.quiver(0,0,0,val1[0],val1[1],val1[2],color="b",length=0.02)
val1 = np.dot(Q_ji,fin[0])
ax.quiver(0,0,0,val1[0],val1[1],val1[2],color="b",length=0.02)
#exit(0)
plt.scatter(0,0,0)
for val1,val2 in zip(start,fin):
       ax.quiver(0,0,0,val1[0],val1[1],val1[2],color="k",length=0.02)
       ax.quiver(0,0,0,val2[0],val2[1],val2[2],color="r",length=0.02)

plt.show()

exit(0)

#
for val,col in zip([75,50,51,52,53,54],colors):
       print(simulations[val])
       sim = fepx_sim("Cube.sim",path=home+simulations[val]+"/Cube.sim")
       step= "28"
       for id in range(500,502):
              stress = sim.get_output("stress",step=step,res="elsets",ids=[id])
              ori = sim.get_output("ori",step=step,res="elsets",ids=[id])
              print("rod ini",ori)
              ori= rod_to_quat(ori)
              print("quat ini",ori)
              ori = ret_to_funda(ori,cub)
              print("quat fin",ori)

              ori= quat_to_rod(ori)
              print("rod fin",ori)
              #exit(0)
              stress_mat= to_matrix(stress)
              eig_val, eig_vect = np.linalg.eig(stress_mat)
              print("eig_val",eig_val)
              eig_val = normalize_vector(eig_val)
              print("eig_val",eig_val)
              pprint(eig_vect)
              ax.scatter(ori[0],ori[1],ori[2],color=col)
              start = ori#[0, 0 ,0]
              for ind,eig in enumerate(eig_vect):
                     ax.quiver(start[0],start[1],start[2],
                            start[0]+eig[0],start[1]+eig[1],start[2]+eig[2]
                            ,length=eig_val[ind]/30,normalize=True,color=col)

ax.axis("off")
ax.view_init(elev=35., azim=45)
plt.show()
plt.savefig("funda_region")
exit(0)

# https://www.geeksforgeeks.org/python-get-the-indices-of-all-occurrences-of-an-element-in-a-list/#
surf1= [ind for ind, elt in enumerate(X) if  elt ==  b[0]]
Vx= [X[i]for i in surf1]
Vy= [Y[i]for i in surf1]
Vz= [Z[i]for i in surf1]
verts = [list(zip(Vx,Vy,Vz))]
#ax.add_collection3d(Poly3DCollection(verts,zsort=min,alpha=0.1))
surf2= [ind for ind, elt in enumerate(X) if  elt ==  -b[0]]

surf3= [ind for ind, elt in enumerate(Y) if  elt ==  b[0]]
surf4= [ind for ind, elt in enumerate(Y) if  elt ==  -b[0]]

surf5= [ind for ind, elt in enumerate(Z) if  elt ==  b[0]]
surf6= [ind for ind, elt in enumerate(Z) if  elt ==  -b[0]]
 
# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1
    

ax = plt.figure().add_subplot(projection='3d')
X= [20.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 10.0, 30.0, 10.0, 20.0, 40.0, 30.0, 40.0, 40.0, 25.0, 30.0, 35.0, 25.0, 35.0, 25.0, 30.0, 60.0, 35.0, 20.0, 40.0, 60.0, 20.0, 40.0, 60.0]
Y= [20.0, 20.0, 10.5, 1.0, 15.25, 15.25, 15.25, 10.5, 10.5, 5.75, 5.75, 20.0, 5.75, 15.25, 5.75, 12.875, 12.875, 12.875, 10.5, 10.5, 8.125, 8.125, 20.0, 8.125, 10.5, 10.5, 10.5, 1.0, 1.0, 1.0]
Z= [22.76535053029903, 38.96497960559887, 66.77662089771968, 95.93421141486863, 28.38618575891917, 15.135746066298575, 18.89683134970845, 42.13047903563476, 12.313994721968864, 56.269231987969555, 32.07046324924786, 55.2219601120633, 12.35786940187457, 41.921945008763586, 28.16061601330233, 10.070378551835649, 13.476236202374565, 23.735327551549496, 11.389019427629599, 21.305869908317952, 13.20425316321665, 11.845200358118927, 100.34872346753596, 20.213927716104187, 18.26818405993052, 30.495051027879345, 74.47058506243454, 54.23429875499636, 24.48021292531605, 29.959116124492354]

# Plot the 3D surface
ax.scatter(X,Y,Z,marker=".")
ax.plot_trisurf(X, Y, Z,alpha=0.3)


exit(0)
# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(X, Y, Z, zdir='z', offset=-20, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
       xlabel='X', ylabel='Y', zlabel='Z')




#### old and broken



array = [b, c, d]
print("input",np.array(array))
arr_funda = []
quats = []
for ori in array:       
       ori= rod_to_quat(ori)
       quats.append(ori)
       arr_funda.append(ret_to_funda(ori,cub,debug=True))
print("vals",np.array(quats))
arr_funda = np.array(arr_funda)
print("output",arr_funda)

exit(0)
ori = b
ax.quiver(0,0,0,ori[0],ori[1],ori[2])

print("rod ini",ori)
ori= rod_to_quat(ori)
print("quat ini",ori)
ori = ret_to_funda(ori,cub)
print("quat fin",ori)
ori= quat_to_rod(ori)
print("rod fin",ori)

exit(0)

ax.quiver(0,0,0,ori[0],ori[1],ori[2])


