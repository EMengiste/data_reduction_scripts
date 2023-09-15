
import matplotlib.pyplot as plt
SIZE=20

plt.rcParams.update({'font.size': SIZE})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['figure.figsize'] = 6,6
#
plt.rc('font', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  #

def plot_stress_strain(ax,stress,strain,labels=True,lw=5,ls="-",col="k",ylim=[0,201],xlim=[1.0e-7,0.5]):
    ax.plot(strain, stress,col,linestyle=ls,linewidth=lw)
    stress = "Stress "
    strain = "Strain"
    x_label = f'{strain} (\%)'
    y_label = f'{stress} (MPa)'
    # Compile labels for the graphs
    plt.ylim(ylim)
    plt.xlim(xlim)
    #ax.legend()
    if labels:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label,labelpad=10)

def svs_real_svs_sim():
    import fepx_sim
    sim = "sample-feedstock"
    path= "/home/etmengiste/jobs/SERDP/g_0_test"
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
    del simulation
    # calculate the yield values
    yield_values = find_yield(stress,strain)
    ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
    stress_off = yield_values["stress_offset"]
    #
    fig, ax = plt.subplots(1, 1)
    plot_stress_strain(ax,stress,strain,lw=1,ylim=[0,500],xlim=[1.0e-7,max(strain)+0.001])
    
    exx= pd.read_csv("/media/schmid_2tb_1/etmengiste/files/research/serdp/serdp_precip_2021/experimental_data/tensile_data/fs2.csv")
    try:
        stress_exp = [float(i) for i in exx["stress"]]
        strain_exp = [float(i) for i in exx["strain"]]
    except:
        stress_exp = [float(i) for i in exx["'stress'"]]
        strain_exp = [float(i) for i in exx["'strain'"]]
    yield_values = find_yield(stress_exp,strain_exp,number=100)
    plot_stress_strain(ax,stress_exp,strain_exp,col="r",lw=1,ylim=[0,500],xlim=[1.0e-7,max(strain)+0.001])
    ax.plot(ystrain,ystress,"k*",ms=20,label="$\sigma_y$="+str(yield_values["y_stress"]))
    ax.plot(strain,stress_off,"ko--",ms=5)
    fig.subplots_adjust(left=0.15, right=0.97,top=0.98,  bottom=0.11, wspace=0.1, hspace=0.1)        
    ax.legend()    
    plt.show()
    