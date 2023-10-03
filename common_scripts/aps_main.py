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

if __name__ == "__main__":
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
    exit(0)