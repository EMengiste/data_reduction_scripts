
from fepx_sim import *
from plotting_tools import *
from tool_box import *
from pre_processing import *

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
    print("time is ",tic)
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
##
def error_function(val1, val2):
    error_calc =  (abs(val1 - val2)/val1)*100
    return error_calc
#
def svs_real_svs_sim(sim,path,real_paths,name="svs",xlim=[1.0e-7,0.22],outdir=""):
    simulation = fepx_sim(sim,path=path+"/"+sim)
    # Check if simulation value is available 
    # if not post process and get
    try:
        stress=simulation.get_output("stress_eq",step="all")
        strain=simulation.get_output("strain_eq",step="all")
    except:
        simulation.post_process(options="-resmesh stress_eq,strain_eq")
        stress=simulation.get_output("stress_eq",step="all")
        strain=simulation.get_output("strain_eq",step="all")
    #
    del simulation
    # calculate the yield values
    yield_values = find_yield(stress,strain)
    ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
    stress_off = yield_values["stress_offset"]
    ystress_sim = ystress
    #
    fig, ax = plt.subplots(1, 1,figsize=[5,7])
    yield_exp = []
    for real_svs in real_paths:
        exx= pd.read_csv(real_svs)
        try:
            stress_exp = [float(i) for i in exx["stress"]]
            strain_exp = [float(i) for i in exx["strain"]]
        except:
            stress_exp = [float(i) for i in exx["'stress'"]]
            strain_exp = [float(i) for i in exx["'strain'"]]
        yield_values = find_yield(stress_exp,strain_exp,number=70)
        yield_exp.append(yield_values["y_stress"])
        plot_stress_strain(ax,stress_exp[:-1000],strain_exp[:-1000],col="r",lw=1,ylim=[0,500],xlim=[1.0e-7,max(strain)+0.001])
    avg_yield = np.mean(yield_exp)
    print(name,avg_yield)
    error = error_function(avg_yield,ystress_sim)[0]
    ax.plot(ystrain,ystress,"k*",ms=20,label=f"error = {error:.03f}\%")
    
    plot_stress_strain(ax,stress,strain,lw=3,ylim=[0,570],xlim=xlim)
    ax.plot(strain,stress_off,"k--",ms=5)
    if "feed_stock" not in name:
        ax.set_yticklabels([])
        ax.set_xlabel("$\\varepsilon$ (-)")
    else:
        ax.set_ylabel("$\sigma$")
 
    fig.subplots_adjust(left=0.2, right=0.97,top=0.97, bottom=0.15, wspace=0.07)
    ax.legend(loc=4)
    plt.savefig(outdir+name)
    ax.cla()

def multi_svs(paths,sims,name_dict="",names=[],destination=".",name="untitled"
              , ylim=[0,200],inset=False,xlim=[1.0e-7,0.025],vm=False,lw=1
              ,normalize=False,show=False,verbose=False):
    #
    fig, ax = plt.subplots(1, 1)
    label_name=""
    y_stress = []
    y_strain=[]
    elt_num = []
    run_times = []
    for sim,path in zip(sims,paths):
        simulation = fepx_sim(sim,path=path+"/"+sim)
        sim = sims.index(sim)

        if names!="":
            label_name = names[sim]
        if verbose:
            simulation.get_mesh_stat()
            num_elts=simulation.elts
            elt_num.append(num_elts)
            #
            simulation.get_runtime(unit="s")
            run_time=simulation.runtime
            run_times.append(run_time)
        # exit(0)
        # Check if simulation value is available 
        # if not post process and get
        try:
            stress=simulation.get_output("stress",step="all")
            strain=simulation.get_output("strain",step="all")
        except:
            simulation.post_process(options="-resmesh stress,strain")
            stress=simulation.get_output("stress",step="all")
            strain=simulation.get_output("strain",step="all")
        #
        # print(strain[-1])
        if vm:
            stress = [von_mises_stress(i)for i in stress]
            strain = [von_mises_stress(i)for i in strain]
        else:
            stress = stress[:,2]
            strain = strain[:,2]
        print(max(strain))
        # calculate the yield values
        yield_values = find_yield(np.array(stress),strain)
        ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
        y_stress.append(ystress)
        y_strain.append(ystrain)
        plot_stress_strain(ax,np.array(stress),strain,col="k",lw=lw,ylim=ylim,xlim=xlim)
        ax.plot(ystrain,ystress,"k"+marker_styles[int(sim)],ms=10,label=label_name)
        # print(ystrain)
        #ax.plot(strain,stress_off,"ko--",ms=5)
        ax.set_ylabel("$\sigma$")
        ax.set_xlabel("$\\varepsilon$")
    ## Adding insets is difficult
            # 

            # x1,x2,y1,y2 = 0.005,0.008,250,275
            # axins = ax.inset_axes(
            #     [0.25, 0.25, 0.47, 0.47],
            #     xlim=(x1, x2), ylim=(y1, y2), xticklabels=[], yticklabels=[])
            # axins.imshow(Z2, extent=extent, origin="lower")
            # ax.indicate_inset_zoom(axins, edgecolor="black")
    # Compile labels for the graphs
    if inset:
        x1,x2,y1,y2 = min(y_strain), max(y_strain), min(y_stress), max(y_stress)
        xlim= [.996*x1,1.005*x2]
        ylim= [.998*y1,1.001*y2]
        name+="_zoomed"
        # # https://stackoverflow.com/a/27512450
        handles, labels = ax.get_legend_handles_labels()
        # labels, handles = zip(*sorted(zip(labels, handles), key=lambda t: t[0]))
        # print(labels)
        ax.legend(handles[::-1], labels[::-1])
    plt.ylim(ylim)
    plt.xlim(xlim)
    fig.subplots_adjust(left=0.15, right=0.95,top=0.98,  bottom=0.11, wspace=0.1, hspace=0.1)        
    if names!="":
        ax.legend(loc=4)  
    
    if show:
        plt.show()
    else:
        # print(y_stress)
        # print("max",max(y_stress))
        print("wrote "+destination+name+"_svs.png")
        print(os.getcwd())
        fig.savefig(destination+name+"_svs.png",dpi=200)
        return y_stress,elt_num,run_times
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
