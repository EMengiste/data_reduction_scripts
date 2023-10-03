
import matplotlib.pyplot as plt
from tool_box import *
SIZE=35

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

def draw_lattice(ax,type="bcc",scale=20,dims=[1,1,1],quat="",angle=0,axis = [1,0,0],lw=5,ec="k",ms=50,offset=np.array([0,0,0])):
    if quat!="":
        angle,axis=quat_to_angle_axis(quat)
    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            for k in range(0,dims[2]):
                #scale up square
                matrix = np.array(angle_axis_to_mat(angle,axis))*scale
                start = np.dot(np.array([i,j,k])+offset,matrix)
                plot_crystal(ax,start=start,type=type,lw=lw,ec=ec,matrix=matrix,ms=ms)
def scalar_multi(arr,s):
    return [a*s for a in arr]
def look_at_single_crystals(ax,offset,scale=20,lw=5,ls="-",spacing=1,ec="k",ms=50,col="k",ylim=[0,201],xlim=[1.0e-7,0.5],quats=True,show=True,debug=False):
    if quats!="":
        quats=[[1,0,0,0],[1,0,0,0],[1,0,0,0]]
        angle1,axis1 = quats[0][0],quats[0][1:]
        angle2,axis2 = quats[1][0],quats[1][1:]
        angle3,axis3 = quats[2][0],quats[2][1:]

    dims = [1,1,1]
    draw_lattice(ax,type="",scale=scale,dims=dims,axis = axis1,lw=lw,ec=ec,ms=ms,angle =angle1,offset=offset)

    offset=np.array([0,0,0])

    dims = [1,1,1]
    draw_lattice(ax,type=type,scale=scale,dims=dims,axis = axis2,lw=lw,ec=ec,ms=ms,angle =angle2,offset=offset)

    offset=np.array([-2.3,0,0])

    dims = [1,1,1]
    draw_lattice(ax,type="fcc",scale=scale,dims=dims,axis = axis3,lw=lw,ec=ec,ms=ms,angle =angle3,offset=offset)


    ele,azi,roll =[10,100,0]
    #ele,azi = vect_to_azim_elev([4,-5,4])
    print(ele,azi)
    print(spacing)
    ax.view_init(elev=ele, azim=azi)
    ax.set_xlim([-spacing,spacing])
    ax.set_ylim([-50,60])
    #ax.set_zlim([-1,60])
    ax.set_aspect("equal")
    ax.set_proj_type("ortho")
    ax.axis("off")

    #exit(0)
def plot_crystal(ax,start=[0,0,0],dim=[1,1,1],type="",lw=5,matrix="",c="k",c2="r",ec="k",ms=20):
    X=scalar_multi([1,0,0,0,0,1,1,1,1],dim[0])
    Y=scalar_multi([1,1,1,0,0,1,0,0,1],dim[1])
    Z=scalar_multi([1,0,1,1,0,1,1,0,0],dim[2])
    if matrix=="":
        matrix = [[1,0,0],[0,1,0],[0,0,1]]
    if type == "bcc":
        X+=scalar_multi([1/2],dim[0])
        Y+=scalar_multi([1/2],dim[1])
        Z+=scalar_multi([1/2],dim[2])
    if type == "fcc":
        X+=scalar_multi([  1,1/2,1/2,1/2,1/2,  0],dim[0])
        Y+=scalar_multi([1/2,  1,1/2,1/2,  0,1/2],dim[1])
        Z+=scalar_multi([1/2,1/2,  1,  0,1/2,1/2],dim[2])

    vects = [np.dot(np.array([x,y,z]),np.array(matrix))+start
                        for x,y,z in zip(X,Y,Z) ]
    X,Y,Z= np.array(vects).T
    ax.scatter(X[:9],Y[:9],Z[:9],c=c,s=ms,ec=ec,lw=lw)
    ax.scatter(X[9:],Y[9:],Z[9:],c=c2,s=ms,ec=ec,lw=lw)

def coordinate_axis(ax,ori,leng = 0.002,offset_text=1.6,
            lw= 4, offset= np.array([0.01,0,-0.0001]), axis_basis = [[1,0,0],[0,1,0],[0,0,1]],
                    xyz_offset = [[-0.0005,-0.0007,0],[0,-0.001,0],[0,-0.001,-0.0004]],
                    sty = "solid",space="",coo_labs=["x","y","z"], fs=60):
    #
    #      defult params need to be fixed for each axis
    debug = True
    debug = False
    if space == "rod_real":
        rod_labs = ["$r_1,x$","$r_2,y$","$r_3,z$"]
    elif space== "real":        
        rod_labs = ["x","y","z"]
    elif space== "real_latex":        
        rod_labs = ["$x$","$y$","$z$"]
    else:
        rod_labs =coo_labs
    start = np.array(ori)+offset
    lab_offset = -0.0002
    ##
    ##     make into function
    ##
    for ind,eig in enumerate(axis_basis):
            print(axis_basis)
            ax.quiver(start[0],start[1],start[2],
                    eig[0],eig[1],eig[2]
                    ,length=leng,normalize=True
                    ,color="k", linestyle = sty,linewidth=lw)
            #
            leng_text=offset_text*leng
            val_txt=(np.array(eig)*(leng_text))+np.array(start)+np.array(xyz_offset[ind])
            ax.text(val_txt[0],val_txt[1],val_txt[2], rod_labs[ind],fontsize=fs, ha='center',va='center',color='k')
            
            if debug:
                    start = np.array([0,0,0])
                    leng = 0.6
                    lab_offset = np.array([0.0005,0.00,0.0007])
                    lw= 5.6
                    val_txt = start(leng+lab_offset)
                    ax.text(val_txt[0],val_txt[1],val_txt[2], rod_labs[ind],fontsize=fs, color='k')
                    ax.quiver(start[0],start[1],start[2],
                        eig[0],eig[1],eig[2]
                        ,length=leng,normalize=True
                        ,color="k", linestyle = sty,linewidth=lw)
                    #
#
##
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
    
def plot_mean_data(NAME,ylims="",y_label="",name="case",
                       unit="",y_ticks ="",y_tick_lables="",debug=False):
    norm =False
    result ={}
    results = pd.read_csv(NAME+".csv").transpose()
    for i in results:
        result[results[i][name]]= [float(results[i]["ystress"])]
        #print(results[i]["case"],result[results[i]["case"]])
    DOMAIN = [ "CUB"]
    DOM = [ "cubic"]
    NSLIP  = ["2","4", "6"]
    #NSLIP =["2"]
    #ANISO  = [,"125", "150", "175", "200", "400"]
    SETS    = ["1", "2", "3", "4", "5"]
    an = ["Iso.", "1.25", "1.50", "1.75", "2.00", "3.00", "4.00"]
    #SETS = ["1"]
    sets    =  ["solid","dotted","dashdot",(0, (3, 5, 1, 5, 1, 5)),(0, (3, 1, 1, 1, 1, 1))]
    ###
    aniso = [100,125, 150, 175, 200, 300, 400]
    differences= {}
    difs2={}

    if debug:
        NSLIP =["2"]
        DOMAIN = ["CUB"]
        DOM = ["cubic"]

    for dom in DOMAIN:
        fig, axs = plt.subplots(1, 3,sharex="col",sharey="row",figsize=( 23,8))
        for slip in NSLIP:
            #
            ax0= axs[NSLIP.index(slip)]
            #ax1= axs[1][NSLIP.index(slip)]
            #
            ax0.set_title(slip+" slip systems strengthened")
            #
            ax0.set_xlim([115,410])
            #ax1.set_xlim([90,410])
            if ylims!="":
                ax0.set_ylim(ylims)
                #ax1.set_ylim([0.95,2.0])
            else:
                ax0.set_ylim([4.6,15.1])
                #ax1.set_ylim([-0.4,10.4])
            #fig, ax = plt.subplots(1, 1,figsize=(10,12))
            for set,line in zip(SETS,sets):
                #
                index1 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_125"]
                index2 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_150"]
                index3 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_175"]
                index4 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_200"]
                index5 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_300"]
                index6 = result["DOM_"+dom+"_NSLIP_"+slip+"_SET_"+set+"_ANISO_400"]

                list1 = [result["DOM_"+dom+"_ISO"][0],index1[0], index2[0],
                    index3[0], index4[0], index5[0],index6[0]]
                
                differences[dom+"_"+slip+"_"+set] = list1
                #print(list)
                #
                if norm == True:
                    list1 = [i/result["DOM_"+dom+"_ISO"][0] for i in list1]
                #### Start of plotting code
                #
                ax0.plot(aniso,list1,"k",lw=2,linestyle=line,
                    label="Set "+str(set))
                #ax1.plot(aniso,list2,"k",lw=2,linestyle=line,
                #    label="Set "+str(set))
                #
                marker_size =130
                #
                for a, l in zip(aniso, list1):
                    ax0.scatter(a, l, marker="o", s=marker_size,edgecolor="k",color= "k")
                    #ax1.scatter(a, l2, marker="o", s=marker_size,edgecolor="k",color= "k")
                #
                ax0.set_xticks(aniso)
                ax0.set_xticklabels(an,rotation=90)

                #ax1.set_xticks(aniso)
                #ax1.set_xticklabels(an,rotation=90)
        if norm and unit!="":
            unit = "(-)"
        axs[0].set_ylabel(y_label,labelpad=25)

        if y_tick_lables !="":
            axs[0].set_yticks(y_ticks)
            axs[0].set_yticklabels(y_tick_lables)
            axs[0].set_ylabel(y_label+unit,labelpad=25)
        ###
        #axs[0].legend()
        #
        # Put title here
        #title = "Deviation from Taylor Hypothesis: \n"+str(DOM[DOMAIN.index(dom)])+" domain, "+str(slip)+" slip systems\n"
        #
        deg = "$^{\circ}$"
        #
        x_label = f'$p$ (-)'
        #y_label = f'Normalized Misorienation ({deg})'
        fig.supxlabel(x_label,fontsize=SIZE)
        fig.subplots_adjust(left=0.09, right=0.98,top=0.9, bottom=0.2, wspace=0.07, hspace=0.1)
        fig.savefig(NAME+"_"+str(DOM[DOMAIN.index(dom)])+"_mean.png",dpi=400)
    