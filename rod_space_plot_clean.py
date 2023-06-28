from ezmethods import *
# Latex interpretation for plots
plt.rcParams.update({'font.size': 55})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.figsize'] = 20,20

# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1
ax = plt.figure().add_subplot(projection='3d')


plot_rod_outline(ax)

remote= "/run/user/1001/gvfs/sftp:host=acmelabpc2.eng.ua.edu,user=etmengiste"
remote=""
home = remote+"/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_system_study/"

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
cub = Cubic_sym_quats()
#
start = 0
sims =[]
sims = [i for i in range(start,start+6)]
for val,col in zip([0,60,51,52,53,54],colors):
       print(simulations[val])
       sim = fepx_sim("Cube.sim",path=home+simulations[val]+"/Cube.sim")
       step= "28"
       max_v_mineigs=[]
       eig_vects = []
       for id in range(500,600):
              print("\n\n-----====",val,id,"=====---\n\n")
              stress = sim.get_output("stress",step=step,res="elsets",ids=[id])
              ori = sim.get_output("ori",step=step,res="elsets",ids=[id])
              print("rod ini",ori)
              ori= rod_to_quat(ori)
              print("quat ini",ori)
              ori = ret_to_funda(ori,cub)
              print("quat fin",ori)

              ori= quat_to_rod(ori)
              #print("rod fin",ori)
              #exit(0)
              stress_mat= np.array(to_matrix(stress))
              print(stress)
              print(stress_mat)
              stress_mat= to_matrix(stress)
              eig_val, eig_vect = np.linalg.eig(stress_mat)
              eig_val, eig_vect = sort_by_vals(eig_val, eig_vect)
              print("eig_val",eig_val)
              max_v_mineigs.append(abs(max(eig_val)/min(eig_val)))
              eig_vects.append(eig_val)
              #print(abs(max(eig_val)/min(eig_val)))
              print("eig_val",eig_val)
              leng = 0.05
              eig_val = normalize_vector([abs(j) for j in eig_val])
              #print("eig_val",eig_val)
              #pprint(eig_vect)
              ax.scatter(ori[0],ori[1],ori[2],color=col)
              start = ori#[0, 0 ,0]
              for ind,eig in enumerate(eig_vect):
                     ax.quiver(start[0],start[1],start[2],
                            start[0]+eig[0],start[1]+eig[1],start[2]+eig[2]
                            ,length=eig_val[ind]*leng,normalize=True,color=col)

       #print("\n\n\nmin==",min(max_v_mineigs))
       #print(max_v_mineigs.index(max(max_v_mineigs)))
       #print(eig_vects[max_v_mineigs.index(max(max_v_mineigs))])
       #print("avg",np.mean(max_v_mineigs))
       #print("std",np.std(max_v_mineigs))
       #print("max==",max(max_v_mineigs))
       #exit(0)
ax.axis("off")
ax.view_init(elev=35., azim=45)
plt.show()
plt.savefig("funda_region")
exit(0)
