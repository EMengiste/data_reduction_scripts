from ezmethods import *
# Latex interpretation for plots
plt.rcParams.update({'font.size': 55})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.figsize'] = 20,20

# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1

fig = plt.figure(figsize=[60,60])
ax = fig.add_subplot(121,projection='3d')

ax2 = fig.add_subplot(122,projection='3d')

ax.set(xlim=(-.6, .6), ylim=(-.6, .6), zlim=(-.6, .6),
       xlabel='X', ylabel='Y', zlabel='Z')

plot_rod_outline(ax)
#plot_rod_outline(ax2)
remote= "/run/user/1001/gvfs/sftp:host=acmelabpc2.eng.ua.edu,user=etmengiste"
#remote=""
home = remote+"/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_system_study/"
rem=""
#home=rem+"/media/schmid_2tb_1/etmengiste/files/slip_system_study/"

simulations = os.listdir(home)
simulations.sort()
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
aniso = [1.25,1.50,1.75,2.00,3.00,4.00]
cub = Cubic_sym_quats()
#
start_sim = 30
sims =[simulations.index("isotropic")]
sims += [i for i in range(start_sim,start_sim+6)]
print([simulations[i] for i in sims])
value=[]
grain_id = 506
sty = "solid"
for val,col,alp in zip(sims[1:],colors[1:],aniso):
       print(simulations[val])
       sim = fepx_sim("Cube.sim",path=home+simulations[val]+"/Cube.sim")
       step= "28"
       max_v_mineigs=[]
       eig_vects = []
       for id in range(500,510):

              leng = 0.06
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
              stress_mat= np.array(to_matrix(stress))
              #print(stress)
              #print(stress_mat)
              stress_mat= to_matrix(stress)
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
              if val == start_sim :
                     #ax2.text(ori[0],ori[1],ori[2], simulations[val],fontsize=10, color='red')
                            
                     ax.text(ori[0],ori[1],ori[2], str(id),fontsize=10, color='k')
              ax.scatter(ori[0],ori[1],ori[2],color=col,edgecolor="k")
              if id == grain_id:
                     ax2.text(ori[0]+0.01,ori[1]+0.01,ori[2]+0.01, simulations[val],fontsize=10, color='red')
                            
                     ax2.scatter(ori[0],ori[1],ori[2],s=200,color=col,edgecolor="k")
                     ax2.text(ori[0],ori[1],ori[2], str(id),fontsize=10, color='red')
              start = ori#[0, 0 ,0]
              for ind,eig in enumerate(eig_vect):
                     ax.quiver(start[0],start[1],start[2],
                            start[0]+eig[0],start[1]+eig[1],start[2]+eig[2]
                            ,length=norm_eig_val[ind]*leng,normalize=True,color="k",alpha=1/alp,
                            linestyle = sty,linewidth=3)

                     if id == grain_id:
                            leng = 0.01
                            ax2.quiver(start[0],start[1],start[2],
                                   start[0]+eig[0],start[1]+eig[1],start[2]+eig[2]
                                   ,length=norm_eig_val[ind]*leng,normalize=True
                                   ,color="k",alpha=1/(2*alp), linestyle = sty,linewidth=3)

       #print("\n\n\nmin==",min(max_v_mineigs))
       #print(max_v_mineigs.index(max(max_v_mineigs)))
       #print(eig_vects[max_v_mineigs.index(max(max_v_mineigs))])
       #print("avg",np.mean(max_v_mineigs))
       #print("std",np.std(max_v_mineigs))
       #print("max==",max(max_v_mineigs))
       #exit(0)

ax.set_aspect("equal")
ax2.set_aspect("equal")
#ax2.set(xlim=(-0.06,0.2), ylim=(-0.04,0.5), zlim=(-0.09,-0.01),
#       xlabel='X', ylabel='Y', zlabel='Z')
ax.axis("off")
ax2.axis("off")
plt.grid(False)
#https://matplotlib.org/stable/gallery/subplots_axes_and_figures/zoom_inset_axes.html

ax.scatter(0.0919,0.5101,-0.814,c="gray",s=10,alpha=0.01)
ax.view_init(elev=35., azim=45)
#ax2.view_init(elev=13., azim=-106,roll=45)
plt.tight_layout()
plt.show()
#plt.savefig("funda_region")
exit(0)
print("\n\n\n_____",ori,"_______\n\n\n")
"""
for ind,val in enumerate(value):
       eig_val, eig_vect,ori = val
       #print(value)
       ax2.scatter(ori[0],ori[1],ori[2],s=20,color=colors[ind%len(colors)])
       print(ori[0],ori[1],ori[2])
       for ind,eig in enumerate(eig_vect):
              ax2.quiver(start[0],start[1],start[2],
                     start[0]+eig[0],start[1]+eig[1],start[2]+eig[2]
                     ,length=eig_val[ind]*leng,color=colors[ind%len(colors)], linewidths=10)
"""
