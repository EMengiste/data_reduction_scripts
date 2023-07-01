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

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


size = 0.4
#ax.set(xlim=(-size, size), ylim=(-size, size), zlim=(-size, size),
#       xlabel='X', ylabel='Y', zlabel='Z')

plot_rod_outline(ax)
#plot_rod_outline(ax2)
remote= "/run/user/1001/gvfs/sftp:host=acmelabpc2.eng.ua.edu,user=etmengiste"
#remote=""
home = remote+"/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_system_study/"
rem=""
home=rem+"/media/schmid_2tb_1/etmengiste/files/slip_system_study/"

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
aniso = [1,1.25,1.50,1.75,2.00,3.00,4.00]
cub = Cubic_sym_quats()
#
start_sim = 30
sims =[simulations.index("isotropic")]
sims += [i for i in range(start_sim,start_sim+6)]
print([simulations[i] for i in sims])
value=[]
grain_id = 548
sty = "solid"
'''
for val,col,alp in zip(sims,colors,aniso):
       print(simulations[val])
       sim = fepx_sim("Cube.sim",path=home+simulations[val]+"/Cube.sim")
       step= "28"
       max_v_mineigs=[]
       eig_vects = []
       id= grain_id
       leng = 0.07
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
       #if val == start_sim and id==grain_id :
              #ax.text(ori[0],ori[1],ori[2], simulations[val],fontsize=10, color='red')
              #ax.scatter(ori[0],ori[1],ori[2],s=2700,color="white",edgecolor="r")
              #ax.scatter(ori[0],ori[1],ori[2],s=3000,color="white",edgecolor="r")
              #ax.text(ori[0],ori[1],ori[2], str(id),fontsize=10, color='k')
       #ax.scatter(ori[0],ori[1],ori[2],s=150,color=col,edgecolor="k")
       #
       ax.scatter(ori[0],ori[1],ori[2],s=200,color=col,edgecolor="k")
       #ax.text(ori[0],ori[1],ori[2], eig_val,fontsize=10, color='red')
       start = ori#[0, 0 ,0]
       for ind,eig in enumerate(eig_vect):
              leng = 0.003
              ax.quiver(start[0],start[1],start[2],
                     start[0]+eig[0],start[1]+eig[1],start[2]+eig[2]
                     ,length=norm_eig_val[ind]*leng,normalize=True
                     ,color="k",alpha=1/alp, linestyle = sty,linewidth=10,
                     edgecolor="k")
       #print("\n\n\nmin==",min(max_v_mineigs))
       #print(max_v_mineigs.index(max(max_v_mineigs)))
       #print(eig_vects[max_v_mineigs.index(max(max_v_mineigs))])
       #print("avg",np.mean(max_v_mineigs))
       #print("std",np.std(max_v_mineigs))
       #print("max==",max(max_v_mineigs))
       #exit(0)
'''

axis_basis = np.array([[1,0,0],[0,1,0],[0,0,1]])

ori = [0,0,1]
print("\n\n\n_____",ori,"_______\n\n\n")
#ori = np.array([1,1,1])
#ori = [0,0,1]
ori = [0,0,0]
ax_bas = []
size = 1
ax.plot(0,0,0,"b*",ms=20)
#offset= np.array([0.001,-0.001,0.001])
start = np.array(ori)
for ind,eig in enumerate(axis_basis.T):
       leng = 0.5
       #vals = normalize_vector([start[0]+eig[0],start[1]+eig[1],start[2]+eig[2]])
       #vals =eig+ normalize_vector(ori-start)
       #print(start,vals)

       ax.quiver(start[0],start[1],start[2],
              start[0]+eig[0],start[1]+eig[1],start[2]+eig[2]
              ,length=leng
              ,color="r", linestyle = sty,linewidth=10)
       
       #d = eucledian_distance(ori,start)
       #vals =eig+normalize_vector(start2-ori)
       #print(start2,vals)
       #ax.quiver(start2[0],start2[1],start2[2],
       #       vals[0],vals[1],vals[2]
       #       ,length=leng
       #       ,color="b", linestyle = sty,linewidth=10)

offset2= np.array([1,1,0.5])
start2 = np.array(ori)+offset2
ax.quiver(start[0],start[1],start[2],
       start2[0],start2[1],start2[2]
       ,color="r", linestyle = sty,linewidth=1)

start= offset2
colors= ["b", "g", "r"]
for ind,eig in enumerate(axis_basis):
       leng = 0.5
       d = eucledian_distance(ori,start)
       vals =eig+normalize_vector(start)
       print(start,vals)
       ax.quiver(start[0],start[1],start[2],
             vals[0],vals[1],vals[2]
              ,length=leng
              ,color=colors[ind], linestyle = sty,linewidth=10)





#ori = [1,-0.009,0.001]
size = 2
ax.set(xlim=(-size+ori[0], size+ori[0]), ylim=(-size+ori[1], size+ori[1]), zlim=(-size+ori[2], size+ori[2]),
       xlabel='X', ylabel='Y', zlabel='Z')


ax.set_aspect("equal")
#ax.axis("off")
plt.grid(False)
#https://matplotlib.org/stable/gallery/mplot3d/projections.html
ax.set_proj_type('ortho')  # FOV = 0 deg
ax.view_init(elev=90, azim=-90,roll=0)
plt.tight_layout()

#exit(0)
show = True
#show = False
if show:
       plt.show()
else:
       #fig.savefig("funda_region")
       fig.savefig("funda_region_zoomed")
exit(0)