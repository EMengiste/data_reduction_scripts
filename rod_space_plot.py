from ezmethods import *
# Latex interpretation for plots
plt.rcParams.update({'font.size': 20})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] =100
plt.rcParams['figure.figsize'] = 23,23
plt.rcParams['axes.edgecolor'] = 'k'

# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
size = 0.4
fs = 90
leng = 0.004
show_axies=True
marker_size=1000
lw= 10
start_sim = 30
sets    =  ["solid","dotted","dashdot",(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))]
aniso = [1,1.25,1.50,1.75,2.00,3.00,4.00]
colors = [[1-1/i,1-1/i,1-1/i] for i in aniso]

destination = "/home/etmengiste/jobs/aps/step27/rod_space_plots"
step= "27"
#ax.set(xlim=(-size, size), ylim=(-size, size), zlim=(-size, size),
#       xlabel='X', ylabel='Y', zlabel='Z')

#plot_rod_outline(ax2)
remote= "/run/user/1001/gvfs/sftp:host=acmelabpc2.eng.ua.edu,user=etmengiste"
#remote=""
home = remote+"/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_system_study/"
rem="/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu"
rem=""
home=rem+"/media/schmid_2tb_1/etmengiste/files/slip_system_study/"

simulations = os.listdir(home)
simulations.sort()
simulations.remove("common_files")

cub = Cubic_sym_quats()
#
sims =[simulations.index("isotropic")]
sims += [i for i in range(start_sim,start_sim+6)]
print([simulations[i] for i in sims])
value=[]

find_grain=False
#find_grain=True

show_circles=False
#show_circles=True
#
grain_of_interest_1 = 592

grain_of_interest_1 = 574
fig_g1 = plt.figure(figsize=(13.5,23))
ax_g1 = fig_g1.add_subplot(projection='3d')
ax_g1.set_title("Grain 1",fontsize=fs)




grain_of_interest_2 = 545
fig_g2 = plt.figure(figsize=(13.5,23))
ax_g2 = fig_g2.add_subplot(projection='3d')
ax_g2.set_title("Grain 2",fontsize=fs)


of_lab_1=[0.5,0 ,0.244] #label offset grain 1
of_lab_2=[-0.15,-0.01 ,-0.112] #label offset grain 2

fig_large = plt.figure(figsize=(27,23))
ax_large = fig_large.add_subplot(projection='3d')
plot_rod_outline(ax_large)
grain_ids=[i for i in range(500,600)]
grain_ids=[grain_of_interest_2,grain_of_interest_1]
for grain_id in grain_ids:#[-60:-40]:
       sty="solid"
       for val,alp,col in zip(sims,aniso,colors):
              print(simulations[val])
              sim = fepx_sim("Cube.sim",path=home+simulations[val]+"/Cube.sim")
              max_v_mineigs=[]
              eig_vects = []
              id= grain_id
              print("-----====",simulations[val],id,"=====---")
              stress = sim.get_output("stress",step=step,res="elsets",ids=[id])
              ori = sim.get_output("ori",step=step,res="elsets",ids=[id])
              #print("rod ini",ori)
              ori = ret_to_funda(rod=ori,sym_operators=cub)
              #print("quat fin",ori)

              ori= quat_to_rod(ori)
              #print("rod fin",ori)
              #exit(0)
              stress_mat= to_matrix(np.array(stress))
              #print(stress)
              #print(stress_mat)
              eig_val, eig_vect = np.linalg.eig(stress_mat)
              eig_val, eig_vect = sort_by_vals(eig_val, eig_vect)

              #print("eig_val",eig_val)
                     
              max_v_mineigs.append(abs(max(eig_val)/min(eig_val)))

              #eig_vects.append(eig_val)
              #print(abs(max(eig_val)/min(eig_val)))
              #print("eig_val",eig_val)
              norm_eig_val = normalize_vector([abs(j) for j in eig_val])

              #print("eig_val",eig_val)
              #pprint(eig_vect)
              if simulations[val]=="isotropic":
                     first=ori
                     if find_grain:
                            ax_large.text(ori[0],ori[1],ori[2], str(grain_id),fontsize=30, color='red')

                     #ax.text(ori[0],ori[1],ori[2], str(id),fontsize=10, color='k')
              #ax.scatter(ori[0],ori[1],ori[2],s=150,color=col,edgecolor="k")
              #
              #ax.text(ori[0],ori[1],ori[2], eig_val,fontsize=10, color='red')
              start = ori#[0, 0 ,0]
              for ind,eig in enumerate(eig_vect):
                     print(start)
                     ax_large.quiver(start[0],start[1],start[2],
                            eig[0],eig[1],eig[2]
                            ,length=leng*15,normalize=True
                            ,color="k",alpha=1/alp, linestyle = sty,linewidth=lw/5,
                            edgecolor="k")
                     if grain_id == grain_of_interest_1:
                            ax_g1.quiver(start[0],start[1],start[2],
                                   eig[0],eig[1],eig[2]
                                   ,length=leng,normalize=True
                                   ,facecolor=col,alpha=1, linestyle = sty,linewidth=lw,
                                   edgecolor=col)
                            
                     if grain_id == grain_of_interest_2:
                            ax_g2.quiver(start[0],start[1],start[2],
                                   eig[0],eig[1],eig[2]
                                   ,length=leng,normalize=True
                                   ,facecolor=col,alpha=1, linestyle = sty,linewidth=lw,
                                   edgecolor=col)
                            
              if grain_id == grain_of_interest_2:  
                     if simulations[val]=="isotropic":
                            ax_large.text(ori[0]+of_lab_2[0],ori[1]+of_lab_2[1],ori[2]+of_lab_2[2],
                                           "Grain 2",fontsize=fs, color='k')
                     ax_g2.scatter(ori[0],ori[1],ori[2],s=marker_size,color="k",edgecolor="k")
              if grain_id == grain_of_interest_1:
                     if simulations[val]=="isotropic":
                            ax_large.text(ori[0]+of_lab_1[0],ori[1]+of_lab_1[1],ori[2]+of_lab_1[2],
                                           "Grain 1",fontsize=fs, color='k')
                     ax_g1.scatter(ori[0],ori[1],ori[2],s=marker_size,color="k",edgecolor="k")
              ax_large.scatter(ori[0],ori[1],ori[2],s=marker_size/80,color="k",edgecolor="k")
              
              #print("\n\n\nmin==",min(max_v_mineigs))
              #print(max_v_mineigs.index(max(max_v_mineigs)))
              #print(eig_vects[max_v_mineigs.index(max(max_v_mineigs))])
              #print("avg",np.mean(max_v_mineigs))
              #print("std",np.std(max_v_mineigs))
              #print("max==",max(max_v_mineigs))
              if val == start_sim and show_circles and (grain_id == grain_of_interest_2 or grain_id==grain_of_interest_2_1) :
                     #ax.text(ori[0],ori[1],ori[2], str(grain_id),fontsize=10, color='red')
                     #ax_large.scatter(ori[0],ori[1],ori[2],s=2700,color="white",edgecolor="r")
                     ax_large.scatter(ori[0],ori[1],ori[2],color="white",edgecolor="r",lw=100,s=200)
                     pass
              #exit(0)


       print("\n\n\n_____",ori,"_______\n\n\n")
       if grain_id == grain_of_interest_2:
              #

              offset= np.array([-0.01,0,-0.003]) # old vals for 545
              offset= np.array([-0.001,-0.00,-0.0233])
              xyz_offset = [[0.003,0.0003,0],[0.0013,0.0003,-0.0001],[.0013,0,0.001]]
              xyz_offset = [[0.002,-0.0001,0],[0,0,0],[0,0,0.0008]]# 
              coordinate_axis(ax_g2,first,coo_labs=["$r_1,z$","$r_2,x$","$r_3,y$"],fs=fs,leng=0.004,offset_text=1.5,offset=offset,xyz_offset=xyz_offset)
              #
              xyz_offset = [[0.05,-0.03,0],[0.04,-0.02,0],[0.08,-0.003,-0.0001]]
              xyz_offset = [[0,0,0],[0,0,0],[0,0,0]]# for 574
              bottom = np.array([0.6,0,-0.6])
              coordinate_axis(ax_large,bottom,coo_labs=["$r_1,z$","$r_2,x$","$r_3,y$"],fs=fs,leng=0.15,offset_text=1.5,xyz_offset =xyz_offset )
              #
       if grain_id ==grain_of_interest_1:

              offset= np.array([-0.002,-0.003,-0.005])
              offset= np.array([-0.011,0.005,-0.02])
              xyz_offset = [[0.0006,-0.0004,0],[0,0,-0.0001],[0.0015,0,0.0009]]# for 592
              xyz_offset = [[0.002,0.0015,0],[-.0,0.001,0],[-0.004,0,0.0045]]# for 574
              coordinate_axis(ax_g1,first,coo_labs=["$r_1,z$","$r_2,x$","$r_3,y$"],fs=fs,leng=0.007,offset_text=1.3,offset=offset,xyz_offset=xyz_offset)
              
              
ax_g2.set_aspect("equal")
ax_g2.axis("off")
ax_g2.margins(0.24)
ax_g1.set_aspect("equal")
ax_g1.axis("off")
ax_g1.margins(0.24)

ax_large.set_aspect("equal")
ax_large.axis("off")

frame=False

if frame:
       size=0.01
       core = [0.16830541, -0.35592484, -0.39711903]
       ax_g1.set_xlim([core[0]-size,core[0]+size])
       ax_g1.set_ylim([core[1]-size,core[1]+size])
       ax_g1.set_zlim([core[2]-size,core[2]+size])
       core = [0.14299445, -0.36123213, -0.40368418]
       ax_g2.set_xlim([core[0]-size,core[0]+size])
       ax_g2.set_ylim([core[1]-size,core[1]+size])
       ax_g2.set_zlim([core[2]-size,core[2]+size])

plt.grid(False)

#ax.set(xlim=(-size+start[0], size+start[0]), ylim=(-size+start[1], size+start[1]), zlim=(-size+start[2], size+start[2]),
#       xlabel='X', ylabel='Y', zlabel='Z')
#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/zoom_inset_axes.html

#ax.set_proj_type('ortho')  # ax2.view_init(elev=45., azim=-160)
#ax.view_init(elev=45., azim=-160)
#ele,azi,roll =[31,-28,0]
ele,azi,roll =[45,76,0]
ax_g2.view_init(elev=ele, azim=azi,roll=roll)
ele,azi,roll =[38,126,0]
ele,azi,roll =[48,144,0]
#ele,azi,roll =[25,155,0]
ax_g1.view_init(elev=ele, azim=azi,roll=roll)
ax_large.view_init(elev=35,azim=45)
plt.tight_layout()
show = True
show = False
if show:
       plt.show()
else:  
       name_1 = "funda_region_zoomed_grain_id_"+str(grain_of_interest_2)+".png"
       name_2 = "funda_region_zoomed_grain_id_"+str(grain_of_interest_1)+".png"
       #fig_large.savefig("funda_region_grain_id_"+str(grain_of_interest_2)+"_grain_id_"+str(grain_of_interest_2_1))
       fig_large.savefig(destination+"/funda_region_clean.png")
       #fig.savefig("funda_region_zoomed")
       fig_g1.savefig(destination+"/"+name_1)
       fig_g2.savefig(destination+"/"+name_2)
       os.chdir(destination)
       os.system("./circle_merge.sh")
exit(0)