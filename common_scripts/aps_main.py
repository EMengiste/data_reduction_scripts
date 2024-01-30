from fepx_sim import *
from plotting_tools import *
from tool_box import *
from pre_processing import *

jobs_path = "/home/etmengiste/jobs/aps/step27/oris/ms_n_t_kasemer"

def reori():
    sample_elts=[505, 513, 545, 553, 561, 593]
    sampled_trajectories(home+"isotropic/Cube.sim/",offset=offset,
                         sampled_elts=sample_elts,sample_dense=sample_dense,debug=False,sim="isotropic",end_path=jobs_path)
    
    #print(tic)
    for sim in simulations[60:66]:
        print(jobs_path)
        sampled_trajectories(home+sim+"/Cube.sim/",debug=False,offset=offset,
                         sampled_elts=sample_elts,sample_dense=sample_dense,sim=sim,end_path=jobs_path)
    os.chdir(jobs_path)
    os.system("./batch_compiled_ori_ipfs.sh")
    exit(0)

def compare_oris(sim1,sim2,destination=""):
    file = open(destination+"misoris.txt","w")
    vals = []
    for ori1,ori2 in zip(sim1,sim2):
        mis = misori(ori1,ori2)
        file.write(str(mis)+"\n")
        vals.append(mis)
    file.close()
    return np.array(vals)

def misori(ori1,ori2):
    axis,angle = rod_to_angle_axis(ori1)
    rotMat1 =angle_axis_to_mat(angle=angle,axis=axis)
    axis,angle = rod_to_angle_axis(ori2)
    rotMat2 =angle_axis_to_mat(angle=angle,axis=axis)
    return dif_degs_local(rotMat1,rotMat2)

def get_yields():
    values = [["name","ystress"]]
    print("starting code")
    tic = time.perf_counter()
    #
    #print(tic)
    for sim in simulations[0:91]:
        print(home+sim)
        if sim!="isotropic":
            sim_num = int(sim)-1
            slip = slips[int((sim_num)/30)]
            set_num = str(int((sim_num%30)/6)+1)
            ani = aniso[int(sim_num%6)]
            name = "DOM_CUB_NSLIP_"+slip+"_SET_"+set_num+"_ANISO_"+ani
            print(name,sim_num)
            #exit(0)
        else:
            name = "DOM_CUB_ISO"
        #print(name)
        simulation = fepx_sim(sim,path=home+sim+"/Cube.sim")
        print(simulation.is_sim)
        # Check if simulation value is available 
        # if not post process and get
        try:
            stress=simulation.get_output("stress",step="malory_archer",comp=2)
            strain=simulation.get_output("strain",step="malory_archer",comp=2)
        except:
            #simulation.post_process(options="-resmesh stress,strain")
            os.chdir(home+sim+"/Cube")
            os.system("neper -S . -resmesh stress,strain")
            os.chdir(home)
            stress=simulation.get_output("stress",step="malory_archer",comp=2)
            strain=simulation.get_output("strain",step="malory_archer",comp=2)
        #exit(0)
        #
        # calculate the yield values
        yield_values = find_yield(np.array(stress),strain)
        print(yield_values["y_stress"])
        values.append([name ,yield_values["y_stress"]])

    
    df1 = pd.DataFrame(values)
    df1.columns=df1.iloc[0]
    df1[1:].to_csv("stresses.csv")
    toc = time.perf_counter()
    print("===")
    print("===")
    print("===")
    print(f"Generated data in {toc - tic:0.4f} seconds")
    #exit(0)
    print("===")
    print("===")
    print("===")
    print("starting plotting")

def yield_streses():
    iso_home="/home/etmengiste/jobs/aps/slip_study/"
    remote = "/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu"
    remote = ""
    home=remote+"/media/schmid_2tb_1/etmengiste/files/slip_system_study/"

    scripts_dir = os.getcwd()
    simulations = os.listdir(home)
    simulations.sort()
    simulations.remove("common_files")
    aniso = ["125", "150", "175", "200", "300", "400"]
    slips = ["2", "4", "6"]
    step = 27
    offset=5
    sample_dense=8
    values = [["name","ystress"]]
    print("starting code")
    #get_yields()
    #exit(0)
    print("===")
    print("===")
    print("===")
    print("starting plotting")
    tic = time.perf_counter()
    plot_mean_data("stresses",y_label="$\sigma_y$",name="name",
                       unit="(MPa)",y_ticks ="",y_tick_lables="",debug=False,ylims=[120,185])
    toc = time.perf_counter()
    print(f"Generated plot in {toc - tic:0.4f} seconds")

if __name__ == "__main__":
    #
    script_fdr=os.getcwd()
    #
    out_path = "/home/etmengiste/jobs/aps/motion_pictures"
    path = "/media/schmid_2tb_1/etmengiste/files/slip_system_study"
    # individual_svs(path,"isotropic/Cube.sim",show=True)
    ids = [i for i in range(499,600,2)]
    oris = np.array([])
    ipf_plotting =  False#True #
    ## initialize sim object
    sim = "isotropic/Cube.sim"
    simulation = fepx_sim(sim,path=path+"/"+sim)
    
    try:
        stress=simulation.get_output("stress",step="all",comp=2)
        strain=100*simulation.get_output("strain",step="all",comp=2)
    except:
        simulation.post_process(options="-resmesh stress,strain")
        stress=simulation.get_output("stress",step="all",comp=2)
        strain=100*simulation.get_output("strain",step="all",comp=2)
    #
    # calculate the yield values
    yield_values = find_yield(stress,strain)
    ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
    stress_off = yield_values["stress_offset"]
    load_steps =len(stress)# 1 # 
    fig, ax = plt.subplots(1, 1)
    fig2 = plt.figure()
    ax3d = fig2.add_subplot(projection="3d")
    os.chdir(out_path)
    for i in range(0,load_steps):
        print(" Plotting step "+str(i))
        if ipf_plotting:
            step_oris = simulation.get_output("ori",step=i,res="elsets",ids=ids)
            oris = np.append(oris,step_oris)
            file_name="ori_"+str(i)
            if i ==0: 
                np.savetxt("ini",oris)
            np.savetxt(file_name,oris)
            os.system(script_fdr+"/bash/ori_ipfs.sh "+str(i)+" >> log_neper")
        # exit(0)
        plot_stress_strain(ax,stress[0:i],strain[0:i],lw=7,ylim=[0,201],xlim=[1.0e-7,5])  
        fig.subplots_adjust(left=0.15, right=0.97,top=0.98,  bottom=0.12, wspace=0.1, hspace=0.1)        
        # plt.show()
        fig.savefig(str(i),dpi=140)
        ax.cla()
    #os.system("./"script_fdr+"/combine_n_animate.sh")
    exit(0)
    sim_1 = fepx_sim("072_g_0_28",path=path+"072_g_0_28.sim")
    path = "/media/schmid_2tb_1/etmengiste/files/research/aps/aps_ori_2022/output_data/slip_system_study/"
    sim_2 = fepx_sim("072",path=path+"072/Cube.sim")

    oris_1 = sim_2.get_output("ori",res="elsets",ids="all",step=27)
    oris_2 = sim_1.get_output("ori",res="elsets",ids="all",step=27)

    misoris = compare_oris(oris_1,oris_2,destination="/home/etmengiste/jobs/aps/aps_add_slip/")
    mean = np.mean(misoris)
    mean = np.std(misoris)
    fig, ax = plt.subplots()
    ax.hist(misoris,bins=1000,label="Mean = "+str(mean)+" Standard deviation = "+str(mean))
    ax.legend()
    plt.show()
    exit(0)