from fepx_sim import *
from plotting_tools import *
from tool_box import *

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
    path = "/home/etmengiste/jobs/aps/aps_add_slip/"
    #individual_svs(path,"072_g_0_28",show=False)
    #exit(0)
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