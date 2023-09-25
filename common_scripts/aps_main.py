from fepx_sim import *
from plotting_tools import *
from tool_box import *

jobs_path = "/home/etmengiste/jobs/aps/step27/oris/ms_n_t_kasemer"



if __name__ == "__main__":
    iso_home="/home/etmengiste/jobs/aps/slip_study/"
    remote = "/run/user/1001/gvfs/sftp:host=schmid.eng.ua.edu"
    remote = ""
    home=remote+"/media/schmid_2tb_1/etmengiste/files/slip_system_study/"

    simulations = os.listdir(home)
    simulations.sort()
    simulations.remove("common_files")
    aniso = ["125", "150", "175", "200", "300", "400"]
    slips = ["2", "4", "6"]
    step = 27
    offset=5
    sample_dense=8

    print("starting code")
    tic = time.perf_counter()
    #
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
    exit(0)