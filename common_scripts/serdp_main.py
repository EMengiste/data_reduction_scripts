from fepx_sim import *
from plotting_tools import *
from tool_box import *

jobs_path = "/home/etmengiste/jobs/SERDP/inhomo_precip/034_precip_09_11_2023"


# precip test with 100 grain
def show_svs_precip_test(path):
    print(path)
    #print(os.listdir(path))
    list_of_dirs = os.listdir(path)
    #print(list_of_dirs)
    list_of_dirs = [i for i in list_of_dirs if not i.endswith(".png")]
    list_of_dirs = [i for i in list_of_dirs if not i.endswith(".sim")]
    list_of_dirs = [i for i in list_of_dirs if not i.endswith(".out")]
    list_of_dirs.remove("images")
    print(list_of_dirs)
    for i in list_of_dirs:
        if i=="feed_stock":
            lables= True
        else:
            lables = False
        simulation = fepx_sim(i,path=path+"/"+i)
        # Check if simulation value is available 
        # if not post process and get
        try:
            stress=simulation.get_output("stress",step="malory_archer",comp=2)
            strain=simulation.get_output("strain",step="malory_archer",comp=2)
        except:
            simulation.post_process(options="-resmesh stress,strain")
            stress=simulation.get_output("stress",step="malory_archer",comp=2)
            strain=simulation.get_output("strain",step="malory_archer",comp=2)
        #
        # calculate the yield values
        yield_values = find_yield(stress,strain)
        ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
        stress_off,index = yield_values["stress_offset"],yield_values["index"]
        #
        fig, ax = plt.subplots(1, 1)
        plot_stress_strain(ax,stress,strain,labels=lables,lw=1,ylim=[0,500],xlim=[1.0e-7,max(strain)+0.001])
        ax.plot(ystrain,ystress,"k*",ms=20,label="$\sigma_y$="+str(yield_values["y_stress"]))
        ax.plot(strain[:index+1],stress_off[:index+1],"ko--",ms=5)
        fig.subplots_adjust(left=0.15, right=0.97,top=0.98,  bottom=0.11, wspace=0.1, hspace=0.1)        
        ax.legend()        
        plt.savefig(path+"/images/"+i+"")
# 
def show_crss_precip_test(path):
    print(path)
    #print(os.listdir(path))
    list_of_dirs = os.listdir(path)
    #print(list_of_dirs)
    list_of_dirs = [i for i in list_of_dirs if not i.endswith(".png")]
    list_of_dirs = [i for i in list_of_dirs if not i.endswith(".sim")]
    list_of_dirs = [i for i in list_of_dirs if not i.endswith(".out")]
    list_of_dirs.remove("images")
    print(list_of_dirs)
    for i in list_of_dirs:
        print(i)

def individual_svs(path,sim):
    simulation = fepx_sim(sim,path=path+"/"+sim)
    # Check if simulation value is available 
    # if not post process and get
    try:
        stress=simulation.get_output("stress",step="malory_archer",comp=2)
        strain=simulation.get_output("strain",step="malory_archer",comp=2)
    except:
        simulation.post_process(options="-resmesh stress,strain")
        stress=simulation.get_output("stress",step="malory_archer",comp=2)
        strain=simulation.get_output("strain",step="malory_archer",comp=2)
    #
    # calculate the yield values
    yield_values = find_yield(stress,strain)
    ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
    stress_off = yield_values["stress_offset"]
    #
    fig, ax = plt.subplots(1, 1)
    plot_stress_strain(ax,stress,strain,lw=1,ylim=[0,500],xlim=[1.0e-7,max(strain)+0.001])
    ax.plot(ystrain,ystress,"k*",ms=20,label="$\sigma_y$="+str(yield_values["y_stress"]))
    ax.plot(strain,stress_off,"ko--",ms=5)
    fig.subplots_adjust(left=0.15, right=0.97,top=0.98,  bottom=0.11, wspace=0.1, hspace=0.1)        
    ax.legend()    
    plt.show()
#
def generate_dense_mesh():
    # generate dense 100 grain mesh
    tesselation_name = "sample"
    print(mesh_loc)
    tess = generate_tess(100,tesselation_name,mesh_loc,options={"mode" :"run"})
    print(tess)
    generate_msh(tess,40,options={"mode" :"remesh","-rcl":"0.1"})

def mesh_density_study(num_mesh,tesselation_name="sample",mode="run"):
    mesh_loc = "/home/etmengiste/jobs/SERDP/dense_mesh"
    # generate dense 100 grain mesh
    print(mesh_loc)
    tess = generate_tess(100,tesselation_name,mesh_loc,options={"mode" :"run"})
    print(tess)
    rcl_increment = 0.85/num_mesh
    print(rcl_increment)
    distributions = []
    names= []
    num_elts = []
    for i in range(num_mesh):
        rcl = str(round(1- (i*rcl_increment),3))
        print(rcl)
        mesh = generate_msh(tess,40,options={"mode" :mode
                                      ,"-rcl": rcl
                                      ,"-statelt":"elsetbody,vol"
                                      ,"-o":"mesh_rcl"+rcl.replace(".","_")})
        file = np.loadtxt(mesh[:-4]+".stelt")
        print(np.shape(file)[0])
        num_elts.append(str(np.shape(file)[0]))
        distributions.append(file)
        names.append("mesh_rcl"+rcl.replace(".","_"))
        print("max depth = ",max(file[:,0]))
    bins = [i for i in range(1,11)]
    fig, ax = plt.subplots(1, 1)
    for dist,name,elts in zip(distributions[::-1],names[::-1],num_elts[::-1]):
        ax.hist(dist[:,0],bins,density=True,label=(name)+" elts="+elts)
    ax.legend()
    plt.show()
    return distributions,names

def generate_crss_from_mask(msh_name="mesh_rcl0_66",mesh_loc = "/home/etmengiste/jobs/SERDP/dense_mesh/"):
    
    file = np.loadtxt(mesh_loc+msh_name+".stelt")
    base = 25
    values = []
    for i in file[:,0]:
        if i>=1:
            values.append(base*2)
        else:
            values.append(base)
    write_crss_file(values,target_dir=mesh_loc,name=msh_name,res="Element")
    print("max depth = ",max(file[:,0]))

def run_test_mesh(msh_name="mesh_rcl0_66",sim_loc="test_1",mesh_loc = "/home/etmengiste/jobs/SERDP/dense_mesh/"):
    script_fdr = os.getcwd()
    os.chdir(mesh_loc)
    os.system("cp simulation.config "+sim_loc+"/.")
    os.system("cp "+msh_name+".msh "+sim_loc+"/simulation.msh")
    os.system("cp "+msh_name+".crss "+sim_loc+"/simulation.crss")
    print(script_fdr+"/serdp_2021_slurm.sh")
    os.chdir(sim_loc)
    os.system(f"sbatch --job-name={msh_name} --hint=nomultithread "+script_fdr+"/serdp_2021_slurm.sh")
    #print(os.listdir("."))


if __name__ == "__main__":
    #show_svs_precip_test(jobs_path)
    mesh_loc = "/home/etmengiste/jobs/SERDP/dense_mesh"
    #mesh_density_study(10,mode="debug")
    #svs_real_svs_sim()
    generate_crss_from_mask(msh_name="mesh_rcl0_235")
    run_test_mesh(msh_name="mesh_rcl0_235",sim_loc="mesh_rcl0_235")
