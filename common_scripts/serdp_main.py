import os
from fepx_sim import *
from plotting_tools import *
from tool_box import *

import time
import multiprocessing 
import os

# jobs_path = "/home/etmengiste/jobs/SERDP/inhomo_precip/034_precip_09_11_2023"



def compare_vm_stress(stress):
    # compare elemental equivalent stress
    stress1,stress2=stress
    vms1 = von_mises_stress(stress1)
    vms2 = von_mises_stress(stress2)
    if vms1==0:
        return 0 
    array = (vms1-vms2)/vms1
    return array

def compare_abs_vm_stress(stress):
    # compare elemental equivalent stress
    stress1,stress2=stress
    vms1 = von_mises_stress(stress1)
    vms2 = von_mises_stress(stress2)
    if vms1==0:
        return 0 
    array = abs(vms1-vms2)/vms1
    return array
def compare_stress_triaxiality(stress):
    #
    # compare elemental equivalent stress
    stress1,stress2=stress.T
    stri1 = stress_triaxiality(stress1)
    stri2 = stress_triaxiality(stress2)
    array = stri2/stri1
    return array

def plot_cummulative_histogram(values,y_label,x_label,xlims,destination,fig_name,suffix,bins_list,histtype="step",value_type="\sigma}_{vm",title=""):
    fig, ax = plt.subplots(1, 1)
    if title=="":
        top=0.98
    else:
        top=0.95
        ax.set_title(title)
    fig.subplots_adjust(left=0.15, right=0.95,top=top,  bottom=0.11, wspace=0.1, hspace=0.1)        
    mean = "{:.3E}".format(np.mean(values))
    std  = "{:.3E}".format(np.std(values))
    ax.hist(values,bins=bins_list,density=True,cumulative=True,ec="k",color="grey"
            ,histtype="step",label="$\\bar{"+value_type+"}=$"+str(mean)+"\n $\\tilde{"+value_type+"}$="+str(std))
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_xlim(xlims)
    ax.set_ylim([0,1.01])
    ax.legend(loc=0)
    fig.savefig(destination+fig_name+suffix)
    ax.cla()   
    plt.close()
    print("=====\n\n\n",mean,std)

def plot_histogram(values,y_label,x_label,xlims,ylims,destination,fig_name,suffix,bins_list,histtype="step",cumulative=False,value_type="\sigma}_{vm",title=""):
    fig, ax = plt.subplots(1, 1)
    if title=="":
        top=0.98
    else:
        top=0.95
        ax.set_title(title)
    fig.subplots_adjust(left=0.15, right=0.96,top=top,  bottom=0.11, wspace=0.1, hspace=0.1)        
    mean = "{:.3E}".format(np.mean(values))
    std  = "{:.3E}".format(np.std(values))
    ax.hist(values,bins=bins_list,density=True,ec="k",color="grey"
            ,label="$\\bar{"+value_type+"}=$"+str(mean)+"\n $\\tilde{"+value_type+"}$="+str(std))
    ax.set_ylabel(y_label)
    ax.set_xlabel(x_label)
    ax.set_xlim(xlims)
    ax.set_ylim(ylims)
    ax.legend(loc=0)
    fig.savefig(destination+fig_name+suffix)
    ax.cla()     
    plt.close()
    print("=====\n\n\n",mean,std)

def compare_ori():
    # compare elemental orientation
    array = [i%53 for i in range(500)]
    return array

def get_interior_elts(stelt):
    file = np.loadtxt(stelt)
    array = [ i for i,val in enumerate(file) if val[1]>0]
    return array

def get_boundary_elts(stelt):
    file = np.loadtxt(stelt)
    array = [ i for i,val in enumerate(file) if val[1]<1]
    return array

def per_elt_comparisions(path,sim,sims,stress_plot=False,strain_pl_plot=True,steps=[2,13,32],destination=""):
    xlims= [0,0.175]
    ylims =[0,50]
    # return histogram 
    print("starting code")
    tic = time.perf_counter()
        #
    print(tic)
    #
    print(sim,sims)
    pool = multiprocessing.Pool(processes=96)
    sim1 = fepx_sim(sim,path=path+"/"+sim)
    sim_steps = sim1.sim_steps
    if steps == "all": 
        steps = [i for i in range(1,sim1.get_num_steps())]
    sim_obj = [fepx_sim(i,path=path+"/"+i) for i in sims[:]]
        
    #oris_homo = sim1.get_output("ori",res="elts",step=step,ids="all")
    #
    #
    #
    #### get the interior and exterior element ids
    stelt_file = "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/output_data/dense_mesh/mesh_rcl0_235.stelt"
    ids =[]
    places = []
    ids.append("all")#[i for i in range(20000)]    
    places.append("")
    ids.append(get_interior_elts(stelt_file))
    places.append("interior")
    ids.append(get_boundary_elts(stelt_file))
    places.append("boundary")
    ###
    #
    #
    for step in steps:#[32]:#[2,8,13,18,22,32]: #range(steps+1):
        print(step)
        for place,ids_truncatedd in zip(places,ids):
            tic1 = time.perf_counter()
            sim_step = sim_steps[step]
            title_val = "$\\varepsilon = "+sim_step+"$"
            suffix   = "_"+place+"_"+str(step)
            y_label = "Density"
            #
            stress_homo=sim1.get_output("stress",res="elts",step=step,ids=ids_truncatedd)
            print(stress_homo[0]) 
            if stress_plot:
                stress_homo=sim1.get_output("stress",res="elts",step=step,ids=ids_truncatedd)
                vm_stress = pool.map(von_mises_stress,stress_homo)
                fig_name = "vm_stress"
                suffix   = "_"+place+sim[:-4]+"_"+str(step)
                x_label = "$\sigma_{vm}$"
                print("max Value",min(vm_stress))
                stress_xlims= [min(vm_stress)*.99,max(vm_stress)*1.01]
                bins_list = np.linspace(stress_xlims[0],stress_xlims[1],1000)
                ylims =[0,0.75]
                #
                #
                plot_cummulative_histogram(vm_stress
                                        ,y_label,x_label,stress_xlims#,ylims
                                        ,destination,fig_name,"_cummulative"+suffix,bins_list
                                        ,histtype="step",title=title_val)
                # plot_histogram(vm_stress
                #                         ,y_label,x_label,stress_xlims,ylims
                #                         ,destination,fig_name,suffix,bins_list
                #                         ,histtype="step",title=title_val)
            #exit(0)          
            if strain_pl_plot: 
                strain_pl_homo=sim1.get_output("strain",res="elts",step=step,ids=ids_truncatedd)
                strain_pl_eq = pool.map(von_mises_stress,strain_pl_homo)
                fig_name = "eq_strain_pl_homogenous"
                x_label = "$E^P$"
                xlims= [0,0.2]
                bins_list = np.linspace(xlims[0],xlims[1],1000)
                ylims =[0,1000]
                #
                #
                plot_cummulative_histogram(strain_pl_eq 
                                        ,y_label,x_label,xlims#,ylims
                                        ,destination,fig_name,"_cummulative"+suffix,bins_list
                                        ,histtype="step",value_type="E}^{p",title=title_val)

                plot_histogram(strain_pl_eq 
                                        ,y_label,x_label,xlims,ylims
                                        ,destination,fig_name,suffix,bins_list
                                        ,histtype="step",value_type="E}^{p",title=title_val)
                
            toc = time.perf_counter()
            print(f"Generated first plot in in {toc - tic1:0.4f} seconds")
            print("\n\n\n======")
            # exit(0)
            # continue
            for i,sim2 in zip(sims,sim_obj):
                #
                tic = time.perf_counter()
                # #
                stress=sim2.get_output("stress",res="elts",step=step,ids=ids_truncatedd)
                vm_stress = pool.map(von_mises_stress,stress)
                # if place ==places[0]:
                vm_values = np.array([stress_homo,stress])
                print("-------++++===",vm_values.shape)
                vm_values= np.moveaxis(vm_values,0,1)
                print("-------++++===",vm_values.shape)
                vm_stress_comparison = pool.map(compare_vm_stress,vm_values)
                fig_name = "vm_stress"
                suffix   = "_"+place+i[:-4]+"_"+str(step)
                xlims =[-0.3,0.5] #[0.99*min(vm_stress_comparison),1.01*max(vm_stress_comparison)]
                bins_list = np.linspace(xlims[0],xlims[1],1000)
                print(40*"--",destination+fig_name+suffix+"comparison")
                np.savetxt(destination+fig_name+suffix+"comparison",vm_stress_comparison)
                print("plotting comparison values")
                plot_cummulative_histogram(vm_stress_comparison
                                        ,y_label,x_label,xlims#,ylims
                                        ,destination,fig_name+"comparison",suffix,bins_list
                                        ,histtype="step",title=title_val)
                
                continue
                exit(0)
                if stress_plot:
                    y_label = "Density"
                    x_label = "$\sigma_{vm}$"
                    xlims= [0,400]
                    bins_list = np.linspace(stress_xlims[0],stress_xlims[1],1000)
                    ylims =[0,0.75]
                    plot_cummulative_histogram(vm_stress
                                            ,y_label,x_label,stress_xlims#,ylims
                                            ,destination,fig_name,"_cummulative"+suffix,bins_list
                                            ,histtype="step",title=title_val)
                    #
                    # plot_histogram(vm_stress
                    #                         ,y_label,x_label,stress_xlims,ylims
                    #                         ,destination,fig_name,suffix,bins_list
                    #                         ,histtype="step",title=title_val)
                    #
                #exit(0)          
                if strain_pl_plot: 
                    strain_pl=sim2.get_output("strain",res="elts",step=step,ids=ids_truncatedd)
                    strain_pl_eq = pool.map(von_mises_stress,strain_pl)
                    fig_name = "eq_strain_pl"
                    suffix   = "_"+place+i[:-4]+"_"+str(step)
                    x_label = "$E^P$"
                    xlims= [0,0.2]
                    bins_list = np.linspace(xlims[0],xlims[1],1000)
                    ylims =[0,600]
                    #
                    #
                    plot_cummulative_histogram(strain_pl_eq 
                                            ,y_label,x_label,xlims#,ylims
                                            ,destination,fig_name,"_cummulative"+suffix,bins_list
                                            ,histtype="step",value_type="E}^{p",title=title_val)

                    plot_histogram(strain_pl_eq 
                                            ,y_label,x_label,xlims,ylims
                                            ,destination,fig_name,suffix,bins_list
                                            ,histtype="step",value_type="E}^{p",title=title_val)
                    
                print(f"Generated first plot in in {toc - tic:0.4f} seconds")
                # #
                # #### stress triaxiality
                # ##
                # a = pool.map(compare_stress_triaxiality,arr)
                # ax.hist(a[1],density=True,ec="k",color="grey",bins=100)
                # ax.set_ylabel("Density")
                # ax.set_xlabel("Normalized triaxiality")
                # fig.savefig("vm_stress_tri_"+place+i[:-4])
                # ax.cla()
                # #
                # tic2 = time.perf_counter()
                # ####
                # toc = time.perf_counter()
                # print(f"Generated second plot in in {toc - tic2:0.4f} seconds")
            #
        ####
    toc = time.perf_counter()
    print("===")
    print("===")
    print(f"Generated data in {toc - tic:0.4f} seconds")
    # do for grain boundry elts
    # do for interior elts
    # return histogram of number of elts
    return

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
#
##
def svs_real_svs_sim(sim,path,real_paths):
    simulation = fepx_sim(sim,path=path+"/"+sim)
    # Check if simulation value is available 
    # if not post process and get
    try:
        stress=simulation.get_output("stress",step="all",comp=2)
        strain=simulation.get_output("strain",step="all",comp=2)
    except:
        simulation.post_process(options="-resmesh stress,strain")
        stress=simulation.get_output("stress",step="all",comp=2)
        strain=simulation.get_output("strain",step="all",comp=2)
    #
    del simulation
    # calculate the yield values
    yield_values = find_yield(stress,strain)
    ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
    stress_off = yield_values["stress_offset"]
    #
    fig, ax = plt.subplots(1, 1,figsize=[5,7])
    for real_svs in real_paths:
        exx= pd.read_csv(real_svs)
        try:
            stress_exp = [float(i) for i in exx["stress"]]
            strain_exp = [float(i) for i in exx["strain"]]
        except:
            stress_exp = [float(i) for i in exx["'stress'"]]
            strain_exp = [float(i) for i in exx["'strain'"]]
        yield_values = find_yield(stress_exp,strain_exp,number=100)
        plot_stress_strain(ax,stress_exp[:-3000],strain_exp[:-3000],col="r",lw=1,ylim=[0,500],xlim=[1.0e-7,max(strain)+0.001])
    ax.plot(ystrain,ystress,"k*",ms=20)
    plot_stress_strain(ax,stress,strain,lw=3,ylim=[0,550],xlim=[1.0e-7,max(strain)+0.001])
    ax.plot(strain,stress_off,"ko--",ms=5)
    if sim !="feed_stock":
        ax.set_ylabel("")
        ax.set_yticklabels([])
 
    fig.subplots_adjust(left=0.2, right=0.97,top=0.97, bottom=0.15, wspace=0.07)
    #ax.legend()   
    plt.savefig(sim)
    ax.cla()

def multi_svs(paths,sims,destination=".",name="untitled",ylim=[0,1.2],xlim=[1.0e-7,0.025],normalize=False,show=False):
    #
    fig, ax = plt.subplots(1, 1)
    ys = []
    for sim,path in zip(sims,paths):
        simulation = fepx_sim(sim,path=path+"/"+sim)
        # Check if simulation value is available 
        # if not post process and get
        try:
            stress=simulation.get_output("stress",step="all",comp=2)
            strain=simulation.get_output("strain",step="all",comp=2)
        except:
            simulation.post_process(options="-resmesh stress,strain")
            stress=simulation.get_output("stress",step="all",comp=2)
            strain=simulation.get_output("strain",step="all",comp=2)
        #
        print(strain[-1])
        # calculate the yield values
        yield_values = find_yield(np.array(stress),strain)
        ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
        ys.append(ystress)
        stress_off = yield_values["stress_offset"]
        plot_stress_strain(ax,np.array(stress),strain,col="k",lw=.1,ylim=ylim,xlim=xlim)
        ax.plot(ystrain,ystress,"k",ms=20)
        print(ystrain)
        #ax.plot(strain,stress_off,"ko--",ms=5)
        ax.set_ylabel("$\sigma$")
        ax.set_xlabel("$\\varepsilon$")
    
    # Compile labels for the graphs
    plt.ylim(ylim)
    plt.xlim(xlim)
    fig.subplots_adjust(left=0.15, right=0.97,top=0.98,  bottom=0.11, wspace=0.1, hspace=0.1)        
    #ax.legend()  
    if show:
        plt.show()
    else:
        print(ys)
        print("max",max(ys))
        print("wrote "+destination+name+"_svs.png")
        fig.savefig(destination+name+"_svs.png",dpi=200)
#
def plot_real_vs_sim_svs():    
    sim = "sample-feedstock"
    sim = "feed_stock"
    path= "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/output_data/grid_search_run/034"
    real_paths = ["/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/tensile_data/fs2.csv",
                  "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/tensile_data/fs3.csv",
                  "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/tensile_data/fs4.csv",
                  "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/tensile_data/fs1.csv"]
    svs_real_svs_sim(sim,path,real_paths)
    #exit(0)
    sim = "sample-bottom"
    sim = "dep_rapid_bottom"
    path= "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/output_data/grid_search_run/034"
    real_paths = ["/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/Overaged Microstructure and Mechanical Data/Tensile Data/svs_sheets/sheet_5.csv",
                  "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/Overaged Microstructure and Mechanical Data/Tensile Data/svs_sheets/sheet_4.csv"]
    svs_real_svs_sim(sim,path,real_paths)
    sim = "sample-deposited"
    sim = "dep_old"
    path= "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/output_data/grid_search_run/034"
    real_paths = ["/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/tensile_data/LD1_mono.csv",
                  "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/tensile_data/LD2_mono.csv",
                  "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/tensile_data/LD3_mono.csv"]
    svs_real_svs_sim(sim,path,real_paths)
    sim = "sample-top"
    sim = "dep_rapid_top"
    path= "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/output_data/grid_search_run/034"
    real_paths = ["/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/Overaged Microstructure and Mechanical Data/Tensile Data/svs_sheets/sheet_1.csv",
                  "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/Overaged Microstructure and Mechanical Data/Tensile Data/svs_sheets/sheet_0.csv"]
    svs_real_svs_sim(sim,path,real_paths)
    exit(0)

def plot_element_volume_distribution(stelt_file):
    file = np.loadtxt(stelt_file)
    file = np.array([ val for i,val in enumerate(file) if val[1]<1])
    volumes=file[:,2]

    # "id,elsetbody,vol,elsetnb,x,y,z"
    fig, ax = plt.subplots(1, 1)
    fig.subplots_adjust(left=0.15, right=0.97,top=0.98,  bottom=0.11, wspace=0.1, hspace=0.1)    
    mean = '{:0.4e}'.format(np.mean(volumes))
    std  = '{:0.4e}'.format( np.std(volumes))
    ax.hist(volumes,density=True,ec="k",color="grey",bins=100,label="Mean ="+str(mean)+"\n Standard deviation="+str(std))
    ax.legend()
    fig.savefig("volumes")

def per_step_elt_val_plotting(path,sim,sims,filename,xlim=[0,0.101],ylim=[-0.2,0.5],steps=[2,13,32]
                              ,destination="",show=False,section="",comp_function = compare_vm_stress):
    print("_+++===per_step_elt_val_plotting()")
    print("starting code")
    tic = time.perf_counter()
        #
    print(tic)
    print(path)
    print(sim)
    print(sims)
    print(destination)
    #
    parameter = "stress"
    res="elts"
    #        #### get the interior and exterior element ids
    stelt_file = "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/output_data/dense_mesh/mesh_rcl0_235.stelt"
    ids = "all"#[i for i in range(20000)]   
    #
    if section=="interior": 
        ids = get_interior_elts(stelt_file)
    if section=="boundary": 
        ids = get_boundary_elts(stelt_file)
        ###
    # ids = ids[:10]
    print(3*"==="+"\n")
    print(3*"==="+"\n")
    print(3*(section+"    ")+"\n")
    print(3*"==="+"\n")
    print(3*"==="+"\n")
    pool = multiprocessing.Pool(processes=multiprocessing.cpu_count())
    sim1 = fepx_sim(sim,path=path+"/"+sim) # reference simulation
    sim_steps = sim1.sim_steps
    if steps == "all": 
        steps = [i for i in range(1,sim1.get_num_steps())]
    sim_obj = {i:fepx_sim(i,path=path+"/"+i) for i in sims[:]}
    values = {}

        ####
    for sim2 in sims: #  set of test cases
        sim2 = sim_obj[sim2]        
        values[sim2]=[]
        for step in steps:#range(1,32):#
            # timer
            # timer
            print("starting step",step)
            tic = time.perf_counter()
            # timer
            # timer
            # print(ids)
            sim1_values = sim1.get_output(parameter,ids=ids,step=step,res=res)            
            sim2_values = sim2.get_output(parameter,ids=ids,step=step,res=res)
            computed_values = np.array([sim1_values,sim2_values])
            # print("-------++++===",computed_values.shape)
            computed_values= np.moveaxis(computed_values,0,1)
            # print("-------++++===",computed_values.shape)
            compared_value = pool.map(comp_function,computed_values)
            mean = np.mean(compared_value)
            std = np.std(compared_value)
            values[sim2].append([float(sim_steps[step]),mean,std])

            # timer
            # timer
            toc = time.perf_counter()
            print("===")
            print("===")
            print(f"Generated data in {toc - tic:0.4f} seconds")
            # timer
            # timer        
        values[sim2] =np.array(values[sim2])
        print("data prepaired plotting")
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(values[sim2][:,0],values[sim2][:,1],"k-*",label="mean")
        ax.plot(values[sim2][:,0],values[sim2][:,2],"r-v",label="std")

        strain='$'
        strain+="\\varepsilon"
        strain+='$'

        x_label = f'{strain}'
        ax.set_xlabel(x_label)
        ax.set_ylabel("$\sigma_{"+"diff"+"}$")
        scale = sim2.name[:-4].split("_")[1]
        # ax.set_title(scale+" "+section)
        ax.set_xlim([0,0.101])
        ax.set_ylim([-0.2,0.5])
        ax.legend()
        fig.subplots_adjust(left=0.17, right=0.95,  bottom=0.11, wspace=0.1, hspace=0.1) 
        if show:
            plt.show()
        fig.savefig(destination+file_name+scale+"_"+section+"comparision_file")

        print(destination+file_name+sim2.name[:-4]+section+"comparision_file")
        np.savetxt(destination+file_name+sim2.name[:-4]+section+"comparision_file"
                   ,values[sim2])
        
def plot_stress_diff_from_file(sims,xlim=[0,0.101],ylim=[0,0.3]
                              ,destination="",show=True,section=""):    
    values = {}

    for sim2 in sims: #  set of test cases
        values[sim2] =np.loadtxt(destination+file_name+sim2[:-4]+section+"comparision_file")
        print("data prepaired plotting")
        fig = plt.figure()
        ax = fig.add_subplot()
        ax.plot(values[sim2][:,0],values[sim2][:,1],"k-*",label="mean")
        ax.plot(values[sim2][:,0],values[sim2][:,2],"r-v",label="std")

        strain='$'
        strain+="\\varepsilon"
        strain+='$'

        x_label = f'{strain}'
        ax.set_xlabel(x_label)
        ax.set_ylabel("$\sigma_{"+"diff"+"}$")
        scale = sim2.split("_")[1]
        # ax.set_title(scale+" "+section)
        ax.set_xlim([0,0.101])
        ax.set_ylim([0,0.3])
        ax.legend()
        fig.subplots_adjust(left=0.17, right=0.95,  bottom=0.11, wspace=0.1, hspace=0.1) 
        if show:
            plt.show()
        fig.savefig(destination+file_name+scale+"_"+section+"comparision_file")
        print(destination+file_name+scale+"_"+section+"comparision_file")

def write_ori_file(ext_file):
    file= open(ext_file,"r").readlines()
    ori_file=open("simulation.ori","w")
    ori_file.write("$ElsetOrientations\n")
    ori_file.write(str(len(file))+" euler-bunge\n")
    for ind,line in enumerate(file):
        ori_file.write(str(ind+1)+" "+line)
    ori_file.write("$EndElsetOrientations")

def job_submission(script_fdr="",sim_set_path="",run=False):
    dir_contents = [i for i in os.listdir() if i.endswith(".msh")]
    ext_file = [i for i in os.listdir() if i.endswith(".ori")]
    sims = []
    rcls = []
    print(dir_contents)
    print(ext_file)
    write_ori_file(ext_file[1])
    for ind,mesh in enumerate(dir_contents[::-1]):
        mesh_name = mesh#[0][:-4]
        print("  ")
        print("==>",mesh_name,"    ",float(mesh_name[8:-4].replace("_",".")))
        rcl=float(mesh_name[8:-4].replace("_","."))
        name = "00"+str(ind)
        # os.mkdir("../"+name[-3:])
        print(os.getcwd())
        os.system("cp "+mesh_name+" ../"+name+"/simulation.msh")
        os.system("cp simulation.ori ../"+name+"/")
        os.system("cp simulation.config ../"+name+"/")
        os.chdir(sim_set_path+"/"+name)
        sims.append(name)
        rcls.append(rcl)
        if run : os.system(f"sbatch --job-name=RCL{rcl} --hint=nomultithread "+script_fdr+"/serdp_2021_slurm.sh") 
        os.chdir(sim_set_path+"/common_files")
    return sims,rcls

if __name__ == "__main__":
    # stress1 = [-11.64722, -12.73054, 143.3881, 6.903182, -28.18609, 31.0733]
    # stress2 = [-11.64722, -12.73054, 143.3881, 6.903182, -28.18609, 31.0733] 
    # stresses = [stress1 for i in range(10)]
    # stresses = np.array([stresses,stresses])
    # stresses = np.moveaxis(stresses,0,1)
    # print(stresses.shape)
    # print(stresses[0])
    # # exit(0)
    # # print(stresses[:,:])
    # # exit(0)
    # pool = multiprocessing.Pool(processes=96)
    # print(pool.map(compare_vm_stress,stresses))
    # exit(0)
    ##   plot element stress distribution
    sim_set_path= "/home/etmengiste/jobs/SERDP/mesh_study"
    path = sim_set_path+"/common_files"
    script_fdr=os.getcwd()
    os.chdir(path)
    print(os.getcwd())
    sims,rcls = job_submission(script_fdr,sim_set_path)
    pprint(sims)
    pprint(rcls)
    paths = [sim_set_path for _ in sims]
    pprint(paths)
    multi_svs(paths,sims,destination=".",name="untitled",ylim=[0,300],xlim=[1.0e-7,0.025],normalize=False,show=False)
    exit(0)
    sims = ["homogenous_rcl0_235.sim","inhomogenous_elt_rcl0_235.sim","inhomogenous_elset_rcl0_235.sim"]
    # multi_svs([path,path,path],sims,destination=path,ylim=[0,160],xlim=[1.0e-7,0.10],normalize=False,show=True)
    # exit(0)
    file_name = "vm_stress_"
    steps ="all"# [2,13,32]# 
    suffix = "comparison"
    # per_elt_comparisions(path,sims[0],sims[1:],steps=steps,strain_pl_plot=False,stress_plot=True,destination=path+"images/")

    dest = "/home/etmengiste/jobs/SERDP/dense_mesh/common_files/figures/abs_"
    # dest = ""
 
    plot_stress_diff_from_file(sims[1:],file_name,show=False,destination=dest,section="")
    plot_stress_diff_from_file(sims[1:],file_name,show=False,destination=dest,section="boundary")
    plot_stress_diff_from_file(sims[1:],file_name,show=False,destination=dest,section="interior")
    exit(0)    
    per_step_elt_val_plotting(path,sims[0],sims[1:],file_name,steps=steps,destination=dest,comp_function=compare_abs_vm_stress,section="")
    per_step_elt_val_plotting(path,sims[0],sims[1:],file_name,steps=steps,destination=dest,comp_function=compare_abs_vm_stress,section="boundary")
    per_step_elt_val_plotting(path,sims[0],sims[1:],file_name,steps=steps,destination=dest,comp_function=compare_abs_vm_stress,section="interior")
    exit(0)   
    ##
    ##   plot element volume distribution

    stelt_file = "/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/output_data/dense_mesh/mesh_rcl0_235.stelt"
    plot_element_volume_distribution(stelt_file)
    exit(0)

    plot_real_vs_sim_svs()
    exit(0)
    path1 = "/home/etmengiste/jobs/SERDP/dense_mesh/equivalent_homogenous"
    path2 = "/home/etmengiste/jobs/SERDP/dense_mesh/layer_1_inv"
    paths = [path1,path2]
    sims = ["mesh_rcl0_235.sim","mesh_rcl0_235.sim"]
    multi_svs(paths,sims,destination=path1)
    exit(0)
    layer = "layer_1_inv"
    layer = "equivalent_homogenous"
    mesh_loc = "/home/etmengiste/jobs/SERDP/dense_mesh/"
    at_work = "work_stuff" not in os.getcwd().split("/")
    submit = False
    processing = True
    print(os.getcwd())
    os.chdir(mesh_loc)        
    dir_contents = [i for i in os.listdir(os.getcwd())
                    if not i.endswith(".sim") and i != "imgs"]
    #print(dir_contents)
    mesh_density_study()
    mesh_content = np.loadtxt("mesh_rcl0_235.stelt")
    # "id,elsetbody,vol,x,y,z"^^
    elt_x = mesh_content[-3]
    elt_y = mesh_content[-2]
    elt_z = mesh_content[-1]

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
                                      ,"-statelt":"id,elsetbody,vol,elsetnb,x,y,z"
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

def post_process_mesh_density(mesh_loc=".",debug=False):
    dir_contents = [[i,get_val_name(i)] for i in os.listdir(mesh_loc)
                    if i.endswith(".msh") and i.startswith("mesh")]
    dir_contents = np.array(dir_contents)
    pprint(dir_contents[:,1])
    dir_contents = np.array(sort_by_vals(dir_contents[:,1],dir_contents[:,0])[::-1]).T
    pprint(dir_contents)
    # get_min_rcl
    test_rcl_msh = [dir_contents[-5]]
    test_rcl_msh = [dir_contents[0]]
    for msh_name in test_rcl_msh:
        sim_loc ="layer_"+str(layer)+"/"+msh_name[0][:-4]
        if debug:
            print(mesh_loc)
            print(test_rcl_msh)
            print(sim_loc)
            print(mesh_loc+"/"+sim_loc+".sim")
            #exit(0)
        sim= fepx_sim(msh_name[0].split(".")[0],path=mesh_loc+sim_loc)
        # post process if necessary
        if not os.path.exists(mesh_loc+"/"+sim_loc+".sim"):
            sim.post_process(options="-reselt crss")
        visualize(mesh_loc+"/"+sim_loc+".sim"
                    ,options={"-step":"0","-dataeltcol":"real:crss"
                            ,"-showelt":'"y>0.5&&elsetbody<=2"',"-showelt1d":'elt3d_shown'
                            ,"mode" :"run"})

# layer thickness study
def run_layer_study(layer=1,debug=False):
    dir_contents = [[i,get_val_name(i)] for i in os.listdir(mesh_loc)
                    if i.endswith(".msh") and i.startswith("mesh")]
    dir_contents = np.array(dir_contents)
    pprint(dir_contents[:,1])
    dir_contents = np.array(sort_by_vals(dir_contents[:,1],dir_contents[:,0])[::-1]).T
    pprint(dir_contents)
    # get_min_rcl
    test_rcl_msh = [dir_contents[-5]]
    for msh_name in test_rcl_msh:
        sim_loc ="layer_"+str(layer)+"/"+msh_name[0][:-4]
        print(mesh_loc+sim_loc)
        try:
            os.mkdir(mesh_loc+sim_loc)
        except (FileExistsError):
            print("directory already exists")
        generate_crss_from_mask(msh_name=msh_name[0][:-4],layer=layer,mesh_loc=mesh_loc)
        run_test_mesh(msh_name=msh_name[0][:-4],mesh_loc=mesh_loc,sim_loc=mesh_loc+sim_loc,debug=False)
    exit(0)

def generate_crss_from_mask(base=25,scale=2,layer=1,column=0,for_col_mask=False,msh_name="mesh_rcl0_66",mesh_loc = "."):
    print(mesh_loc+msh_name)
    file = np.loadtxt(mesh_loc+msh_name+".stelt")
    values = []
    for i,j in enumerate(file[:,column]):
        if j>=layer:
            values.append(base*scale)
        else:
            values.append(base)
    if for_col_mask:
        np.savetxt(mesh_loc+msh_name+"crss",values,delimiter="\n")
    write_crss_file(values,target_dir=mesh_loc,name=msh_name,res="Element")
    print("max depth = ",max(file[:,column]))

def configure_run_script(nodes=1,processors=48):
    print(nodes,processors)

def run_test_mesh(msh_name="mesh_rcl0_66",script_fdr = os.getcwd(),sim_loc="test_1",mesh_loc = ".",debug=True):
    
    print("---run  mesh---")
    os.chdir(mesh_loc)
    if not os.path.isdir(sim_loc):
        print("directory doesnt exist making")
        os.mkdir(sim_loc)
    os.system("cp simulation.config "+sim_loc+"/.")
    os.system("cp "+msh_name+".msh "+sim_loc+"/simulation.msh")
    os.system("cp "+msh_name+".crss "+sim_loc+"/simulation.crss")

    print(script_fdr+"/serdp_2021_slurm.sh")

    if debug:
        print(mesh_loc)
        print(sim_loc)
        exit(0)
    os.chdir(sim_loc)
    msh_name = msh_name.split("rcl")[1]
    # check if  sim is done20
    
    os.system(f"sbatch --job-name={msh_name} --hint=nomultithread "+script_fdr+"/serdp_2021_slurm.sh")
    #print(os.listdir("."))

def soln_func_1(offset=np.array([0.5,0.5,0.5]),scale=0.002):
    X = np.arange(0, 1, scale)
    Y = np.arange(0, 1, scale)
    X, Y = np.meshgrid(X, Y)
    Z = np.sqrt(((X-offset[0])**2 +(Y-offset[1])**2))
    return X,Y,Z

def soln_func_2(a,b,offset=np.array([0.5,0.5,0.5]),scale=0.002):
    X = np.arange(0, 1, scale)
    Y = np.arange(0, 1, scale)
    X, Y = np.meshgrid(X, Y)
    Z = a*np.sin(X-offset[0]) + b *np.sin(Y-offset[1])
    return X,Y,Z

def soln_func_3(a,b,offset=np.array([0.5,0.5,0.5]),scale=0.002):
    X = np.arange(-0.5, 0.5, scale)
    Y = np.arange(-0.5, 0.5, scale)
    X, Y = np.meshgrid(X, Y)
    R = np.sqrt((X)**2 +(Y)**2)
    Z = np.sin(a*R)
    return X,Y,Z

def mesh_analogue():
    print("start")
    '''    
    ms=200
    lw=3
    axis = [1,0,0]
    angle = 0
    ec="k"
    type="fcc"
    scale=60
    spacing=scale+width    
    width=50
    plot_crystals=False'''
    from matplotlib import cm
    from matplotlib.ticker import LinearLocator
    # https://matplotlib.org/stable/gallery/mplot3d/surface3d.html
    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})

    fig.subplots_adjust(left=0.20)
    offset=np.array([0.5,0.5,0.5])
    scale= 0.0025
    # Make data.
    #X,Y,Z =soln_func_1(offset,scale)
    # make function to make distribution of precipitates within the sample for a given slip system (assume spherical)
    # descritize by the grain sizeand centroid 
    # that is assign elements in the grain strength on a given slip system based on their proximity 
    # to the center of the grain
    # should make shape with same volume in the grain interior to exterior

    X,Y,Z =soln_func_3(1,1,offset,scale)

    # Plot the surface.
    # colormap
    #surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
    #                    linewidth=0, antialiased=False)
    #   testing
    surf = ax.plot_surface(X+offset[0], Y+offset[1], Z+offset[2],
                        linewidth=0, antialiased=False)

    plot_crystal(ax=ax)
    # Customize the z axis.
    ax.set_zlim(-1.01, 1.01)
    ax.zaxis.set_major_locator(LinearLocator(10))
    # A StrMethodFormatter is used automatically
    ax.zaxis.set_major_formatter('{x:.02f}')

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)
    show = True
    if show:
        plt.show()
    else:
        #fig.savefig("funda_region")
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
        bbox = fig.bbox_inches.from_bounds(1, 1, 6, 6)

def get_val_name(var):
    # input: "mesh_rcl#_#.msh"
    # output: "#.#"
    #
    ans = var.split("rcl")[1].split(".")[0].replace("_",".")
    #print(float(ans))
    return float(ans)

def job_submission_1():
    for mesh in dir_contents[::-1]:
        mesh_name = mesh[0][:-4]
        #====::: Generate,process,remesh mesh
        #mesh_density_study(10,mode="stat")
        #
        generate_msh(mesh_loc+mesh_name,48,source_code="neper",options={"mode" :"remesh"})

        #====::: Generate crss distribution
        generate_crss_from_mask(scale=0.5,base=50,column=1,msh_name=mesh_name,
                                for_col_mask=False,layer=1,mesh_loc=mesh_loc)
        #
        #====::: Submit simulation
        run_test_mesh( mesh_name,sim_loc="layer_1/"+mesh_name,mesh_loc=mesh_loc,debug=False)
        #
    exit(0)

def processing_mod():
    preamble = "\makeDataslide{"
    files_processed = ""
    os.chdir(mesh_loc+layer) 
    for i in dir_contents[1:]:
        rcl_val = str(get_val_name(i)).split(".")
        files_processed+=preamble+rcl_val[0]+"}{"+rcl_val[1]+"}\n"
        print(files_processed)
        sim = fepx_sim(i,path=os.getcwd()+"/"+i)
        print(sim)
        if sim.completed:
            #sim.post_process(options="-reselt crss")   
            sim.post_process()           
                
        individual_svs(mesh_loc+layer,i)
        os.chdir(mesh_loc+layer)  

        os.system("neper -V "+i+".sim -step 0 -showelt 'y>=0.5' -showelt1d elt3d_shown -dataeltcol crss -dataeltscale 25:100 -print imgs/"+i+"_step0")
        os.system("neper -V "+i+".sim -step 16 -showelt 'y>=0.5' -showelt1d elt3d_shown -dataeltcol crss -dataeltscale 25:100 -print imgs/"+i+"_step16")
    print(files_processed)
    exit(0)
#
def otherstuff():
    sims_loc = "/home/etmengiste/jobs/SERDP/inhomo_precip/same_macro_different_micro/"
    os.chdir(sims_loc)
    print(os.listdir("."))
    sims = [i for i in os.listdir(".") if i.endswith(".sim")]
    print(sims)
    exit(0)
    multi_svs(sims_loc,sims[::-1],destination=sims_loc)
    exit(0) 
#
def other_other_stuff():
    #mesh_density_study(10,mode="debug")
    exit(0)
    pprint(dir_contents[0:])
    #svs_real_svs_sim()
    #generate_crss_from_mask(column=1,msh_name="mesh_rcl0_235",for_col_mask=True,layer=3,mesh_loc=mesh_loc)
    #post_process_mesh_density(mesh_loc=mesh_loc,debug=True)
    #
    for mesh in dir_contents[::-1]:
        mesh_name = mesh
        if submit:
            run_test_mesh( mesh_name,sim_loc="layer_1_inv/"+mesh_name,mesh_loc=mesh_loc,debug=False)
        #   
            continue
        generate_msh(mesh_loc+mesh_name,48,source_code="neper"
                        ,options={"mode" :"stat"
                                ,"-statelt":"id,elsetbody,vol,x,y,z"})

        #====::: Generate crss distribution
        generate_crss_from_mask(scale=2,base=25,column=1,msh_name=mesh_name,
                                for_col_mask=True,layer=1,mesh_loc=mesh_loc)
        #(input_source,source_code="neper",options={"mode" :"run"})
        visualize(mesh_loc+mesh_name+".msh",source_code="neper",options={"mode" :"run"
                                                                            ,"-dataeltcol":"'real:file("+mesh_name+"crss)'"
                                                                            ,"-dataeltscale":"25:100"
                                                                            ,"-showelt":"'y>=0.5'"
                                                                            ,"-showelt1d":"elt3d_shown"
                                                                            ,"-outdir":"images"})
        #====::: Submit simulation

        #exit(0)
        #run_test_mesh( mesh_name,sim_loc="layer_1/"+mesh_name,mesh_loc=mesh_loc,debug=False)
    #
    exit(0)
    #====::: Post_process
    #  neper -S
    #
#
def post_process_one():
    #show_svs_precip_test(jobs_path)
    layer = "layer_1_inv"
    layer = "equivalent_homogenous"
    mesh_loc = "/home/etmengiste/jobs/SERDP/dense_mesh/"
    at_work = "work_stuff" not in os.getcwd().split("/")
    submit = False
    processing = True
    print(os.getcwd())
    os.chdir(mesh_loc+layer)        
    dir_contents = [i for i in os.listdir(os.getcwd())
                    if not i.endswith(".sim") and i != "imgs"]
    #os.chdir(mesh_loc)  
    other_stuff = False   
    i = os.listdir(os.getcwd())[0]
    individual_svs(mesh_loc+layer,i) 
    os.system("neper -V "+i+".sim -step 0 -showelt 'y>=0.5' -showelt1d elt3d_shown -dataeltcol crss -dataeltscale 25:100 -print imgs/"+i+"_step0")
    os.system("neper -V "+i+".sim -step 16 -showelt 'y>=0.5' -showelt1d elt3d_shown -dataeltcol crss -dataeltscale 25:100 -print imgs/"+i+"_step16")


'''    else:
        mesh_analogue()
        print(os.getcwd())'''

