
import time
import multiprocessing 
from tool_box import *
from pre_processing import *
from plotting_tools import *

def strength_function(inputs):
    dist,base_strength,vars=inputs
    if vars == 1:
        strength = base_strength*dist
    elif vars == 2:
        strength = (base_strength*dist)**2+(dist)
    elif vars == 3:
        strength = (base_strength+ dist*100)
    else:
        strength = vars*dist +base_strength
    return strength

def strength_function(inputs,vars=""):
    dist,base_strength,vars=inputs
    if vars == "":
        strength = (2*base_strength*dist)**2
    else:
        strength = vars*dist +base_strength
    return strength

def generate_strength_files(adjustment=0,base_strength = 25,vis=False,val=1):
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())

    mesh = generate_msh("./simulation",48,options={"mode" : "stat"
                                    ,"-statelt3d":"id,elsetbody,vol,elset3d,x,y,z"
                                    ,"-statelset":"id,vol,x,y,z"
                                    ,"-o":"simulation"})
    #
    # "id,elsetbody,vol,elset3d,x,y,z"
    elt_data = np.loadtxt(mesh+".stelt3d")
    # "id,vol,x,y,z"
    grain_data = np.loadtxt(mesh+".stelset")
    
    strength_values1=[]
    strength_values2=[]
    distances = []
    for i in elt_data:
        elset_id = int(i[-4])
        coos = i[-3:]
        dist = euclidian_distance(grain_data[elset_id-1,-3:],coos)
        distances.append(dist)


    distances = np.array(distances)
    vols = elt_data[:,2]
    
    max_dist = max(distances)
    num_elts = len(distances)
    
    strength_values1 =pool.map(strength_function,np.array([distances,np.tile(base_strength,num_elts)
                        ,np.tile(val,num_elts)]).T)
    # strength_values2 =pool.map(strength_function,np.array([distances-max_dist,np.tile(base_strength+adjustment,num_elts)
                        # ,np.tile(-base_strength,num_elts)]).T)
    

    # ### volume weighting#### ----- debugging for adjustment value
    # strength_values1_vol = [vol*strength for strength,vol in zip(strength_values1,vols)]
    # strength_values2_vol = [vol*strength for strength,vol in zip(strength_values2,vols)]
    # print("Sample 1 "+'{:0.6e}'.format(sum(strength_values1_vol)))
    # print("Sample 2 "+'{:0.6e}'.format(sum(strength_values2_vol)))
    # print("Error ",error_function(sum(strength_values1_vol),sum(strength_values2_vol)))
    # exit(0)

    strength_values1 = np.array(strength_values1)
    # strength_values2 = np.array(strength_values2)

    ### volume weighting
    strength_values1_vol = [vol*strength for strength,vol in zip(strength_values1,vols)]
    # strength_values2_vol = [vol*strength for strength,vol in zip(strength_values2,vols)]
    print("Sample 1 "+'{:0.3e}'.format(sum(strength_values1_vol)))
    # print("Sample 2 "+'{:0.3e}'.format(sum(strength_values2_vol)))
    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # ax.plot(distances,strength_values1,"k.-",label="Sample 1 $\\tau^{vol}_0$ = "+'{:0.3e}'.format(sum(strength_values1_vol)))
    # ax.plot(distances,strength_values2,"r.-",label="Sample 2 $\\tau^{vol}_0$ = "+'{:0.3e}'.format(sum(strength_values2_vol)))
    # ax.hlines(sum(strength_values1_vol),min(distances),max(distances),colors="k")
    # ax.hlines(sum(strength_values2_vol),min(distances),max(distances),colors="r")
    # ax.legend()
    # ax.set_xlabel("d (-)")
    # ax.set_ylabel("$\\tau_0$ (MPa)")
    # fig.subplots_adjust(left=.15,right=0.95,top=0.98, bottom=0.12)   
    # fig.savefig("crss_vs_d",dpi=300)
    # exit(0)
    write_crss_file(strength_values1,target_dir="",name="simulation_001",res="Element")
    # write_crss_file(strength_values2,target_dir="",name="simulation_002",res="Element")
    # exit(0)
    #plot3d_scatter_heat(elt_data[-100:,-3:].T,strength_values[-100:],label='Name [units]')
    np.savetxt("distances"+str(val),distances)
    np.savetxt("strength_mask_pos"+str(val),strength_values1)    
    # np.savetxt("strength_mask_neg",strength_values2)    
    # plt.show()    
    if vis:
        visualize("./simulation.msh",overwrite_file=True
                    ,options={"-dataeltcol":"'real:file(strength_mask_pos"+str(val)+")'"#,"-dataeltcolscheme":"viridis"
                            ,"-showelt":'"y>0.5"',"-showelt1d":'elt3d_shown'
                            ,"-dataeltscale":" 25:55"
                            ,"mode" :"run"},outname="increasing"+str(val))
        # visualize("./simulation.msh",overwrite_file=True
        #             ,options={"-dataeltcol":"'real:file(strength_mask_neg)'"#,"-dataeltcolscheme":"viridis"
        #                     ,"-showelt":'"y>0.5"',"-showelt1d":'elt3d_shown'
        #                     ,"-dataeltscale":" 35:55"
        #                     ,"mode" :"run"},outname="decreasing")

def linear_gradient_run():
    #  UAH colaboration Linear gradient run
    project_home= "/home/etmengiste/jobs/UAH_collaboration/" 
    script_fdr=os.getcwd()


    # linear strength increase for grain
    sim_set_path= project_home+"linear_gradient"
    neper_path = "/home/etmengiste/code/neper/neper-dev/build/neper"
    fepx_path = "/home/etmengiste/bin/fepx_general_input"
    path = sim_set_path+"/common_files"
    os.chdir(path)
    print(os.getcwd())
    adjustment = -4.143 #-2.358 # tau = 40
    pool = multiprocessing.Pool(processes=os.cpu_count())
    # generate_strength_files(base_strength=40,adjustment=adjustment,vis=True)

    # exit(0)
    sims = [1,2]
    for sim in sims:
        sim = ("000"+str(sim))[-3:]
        print(sim)
        os.mkdir("../"+sim)
        os.chdir("../"+sim)
        os.system("rm output.*")
        os.system("rm error.*")
        cwd = os.getcwd()
        print(cwd)
        shutil.copy2(path+"/simulation.msh",cwd+"/simulation.msh")
        shutil.copy2(path+"/simulation.cfg",cwd)
        shutil.copy2(path+"/simulation_"+sim+".crss",cwd+"/simulation.opt")
        job_submission_script(cwd,fepx_path=fepx_path)
        print(os.listdir())
        
    exit(0)


def uah_scripts():
    path = "/home/etmengiste/jobs/UAH_collaboration/linear_gradient"
    paths = [path,path]
    sims = ["001/simulation.sim","002/simulation.sim"]
    multi_svs(paths,sims,vm=False,lw=1,destination="",names=sims, ylim=[0,175]
              ,xlim=[-1.0e-2,0.15],normalize=False,show=True,name="labeled")

    exit(0)
    outdir="/home/etmengiste/jobs/SERDP/AA7050/imgs/"
    # plot_100_mesh_data_precip(outdir=outdir)
    # plot_2000_mesh_data_precip(outdir=outdir)
    # plot_2000_mesh_data_g_0(outdir=outdir)
    exit(0)
    sims = ["00"+str(i+1)+"/simulation.sim" for i in range(8)]
    paths = [path for i in range(8)]
    multi_svs(paths,sims,vm=True,lw=1,destination="",names=sims
              ,xlim=[-1.0e-2,0.70],normalize=False,name="labeled")
    exit(0)

if __name__ == "__main__":
    dest = "/home/etmengiste/jobs/SERDP/test_crss"
    os.chdir(dest)
    generate_strength_files(base_strength=25,vis=True,val=1)
    generate_strength_files(base_strength=25,vis=True,val=2)
    generate_strength_files(base_strength=25,vis=True,val=3)