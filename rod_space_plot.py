from ezmethods import *
# Latex interpretation for plots
plt.rcParams.update({'font.size': 20})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.figsize'] = 23,23
plt.rcParams['axes.edgecolor'] = 'k'

# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1

#fig = plt.figure()
#ax = fig.add_subplot(projection='3d')
size = 0.4
#ax.set(xlim=(-size, size), ylim=(-size, size), zlim=(-size, size),
#       xlabel='X', ylabel='Y', zlabel='Z')

#plot_rod_outline(ax2)
remote= "/run/user/1001/gvfs/sftp:host=acmelabpc2.eng.ua.edu,user=etmengiste"
#remote=""
home = remote+"/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_system_study/"
rem=""
home=rem+"/media/schmid_2tb_1/etmengiste/files/slip_system_study/"

simulations = os.listdir(home)
simulations.sort()
show_axies=True
marker_size=1000
simulations.remove("common_files")
colors = ["k",
             (10/255,10/255,10/255),
             (70/255,70/255,70/255),
             (150/255,150/255,150/255),
             (200/255,200/255,200/255),
             (215/255,215/255,215/255),
             (255/255,255/255,255/255)]
#
sets    =  ["solid","dotted","dashdot",(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))]
aniso = [1,1.25,1.50,1.75,2.00,3.00,4.00]
cub = Cubic_sym_quats()
#
start_sim = 30
sims =[simulations.index("isotropic")]
sims += [i for i in range(start_sim,start_sim+6)]
print([simulations[i] for i in sims])
value=[]
fig = plt.figure()
ax = fig.add_subplot(projection='3d')
#plot_rod_outline(ax)
for grain_id in [585]:
       sty = "solid"
       for val,col,alp in zip(sims,colors,aniso):
              print(simulations[val])

              sim = fepx_sim("Cube.sim",path=home+simulations[val]+"/Cube.sim")
              step= "28"
              max_v_mineigs=[]
              eig_vects = []
              id= grain_id
              leng = 20
              print("-----====",simulations[val],id,"=====---")
              stress = sim.get_output("stress",step=step,res="elsets",ids=[id])
              ori = sim.get_output("ori",step=step,res="elsets",ids=[id])
              #print("rod ini",ori)
              ori= rod_to_quat(ori)
              #print("quat ini",ori)
              ori = ret_to_funda(ori,cub)
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
              #if val == start_sim and id==grain_id :
                     #ax.text(ori[0],ori[1],ori[2], simulations[val],fontsize=10, color='red')
                     #ax.scatter(ori[0],ori[1],ori[2],s=2700,color="white",edgecolor="r")
                     #ax.scatter(ori[0],ori[1],ori[2],s=3000,color="white",edgecolor="r")
                     #ax.text(ori[0],ori[1],ori[2], str(id),fontsize=10, color='k')
              #ax.scatter(ori[0],ori[1],ori[2],s=150,color=col,edgecolor="k")
              #
              #ax.text(ori[0],ori[1],ori[2], eig_val,fontsize=10, color='red')
              start = ori#[0, 0 ,0]
              leng = 0.003
              for ind,eig in enumerate(eig_vect):
                     ax.quiver(start[0],start[1],start[2],
                            eig[0],eig[1],eig[2]
                            ,length=leng,normalize=True
                            ,color="k",alpha=1/alp, linestyle = sty,linewidth=10,
                            edgecolor="k")
              ax.scatter(ori[0],ori[1],ori[2],s=marker_size,color="k",edgecolor="k")
              #print("\n\n\nmin==",min(max_v_mineigs))
              #print(max_v_mineigs.index(max(max_v_mineigs)))
              #print(eig_vects[max_v_mineigs.index(max(max_v_mineigs))])
              #print("avg",np.mean(max_v_mineigs))
              #print("std",np.std(max_v_mineigs))
              #print("max==",max(max_v_mineigs))
              #exit(0)


       print("\n\n\n_____",ori,"_______\n\n\n")
       ori = first

       debug = True
       debug = False
       rod_labs = ["$r_1 , x$","$r_2 , y$","$r_3, z$"]
       xyz_labs = ["$x$","$y$","$z$"]
       axis_basis = [[1,0,0],[0,1,0],[0,0,1]]
       offset= np.array([0.01,0,-0.0001])
       start = np.array(ori)+offset
       fs=60
       ##
       ##     make into function
       ##
       if show_axies:
              for ind,eig in enumerate(axis_basis):

                     leng = 0.002
                     lab_offset = np.array([0.0004,0.0004,0.0005])
                     lw= 4
                     ax.quiver(start[0],start[1],start[2],
                            eig[0],eig[1],eig[2]
                            ,length=leng,normalize=True
                            ,color="k", linestyle = sty,linewidth=lw)
                     #
                     leng*=1.6
                     val_txt=(np.array(eig)*leng)+np.array(start)
                     ax.text(val_txt[0],val_txt[1],val_txt[2], rod_labs[ind],fontsize=fs, color='k')
                     
                     if debug:
                            start = np.array([0,0,0])
                            leng = 0.6
                            lab_offset = np.array([0.0005,0.00,0.0007])
                            lw= 5.6
                            val_txt = start+val*(leng+lab_offset)
                            ax.text(val_txt[0],val_txt[1],val_txt[2], rod_labs[ind],fontsize=fs, color='k')
                            ax.quiver(start[0],start[1],start[2],
                                   eig[0],eig[1],eig[2]
                                   ,length=leng,normalize=True
                                   ,color="k", linestyle = sty,linewidth=lw)
                            #
                     #
                     #
       ax.set_aspect("equal")
       ax.axis("off")
       plt.grid(False)

       size = 0.007
       #ax.set(xlim=(-size+start[0], size+start[0]), ylim=(-size+start[1], size+start[1]), zlim=(-size+start[2], size+start[2]),
       #       xlabel='X', ylabel='Y', zlabel='Z')
       #https://matplotlib.org/stable/gallery/subplots_axes_and_figures/zoom_inset_axes.html

       #ax.set_proj_type('ortho')  # ax2.view_init(elev=45., azim=-160)
       #ax.view_init(elev=45., azim=-160)
       ele,azi,roll =[31,-28,0]
       #ele,azi,roll =[26,62,0]
       ax.view_init(elev=ele, azim=azi,roll=roll)

show = True
show = False
if show:
       plt.show()
else:
       #fig.savefig("funda_region")
       fig.savefig("funda_region_zoomed")
       #fig.savefig("funda_region_zoomed_grain_id_"+str(grain_id))
exit(0)