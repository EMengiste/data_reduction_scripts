from tool_box import *
from fepx_sim import *
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
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

marker_styles = ['o','*','x','X','+','P','s','D','p','v']
def plot3d_scatter_heat(coo,val,label='Name [units]'):
    fig= plt.figure()
    ax = fig.add_subplot(111,projection="3d")
    X,Y,Z=coo
    C = ax.scatter(X,Y,Z,s=val,c="b")
    fig.colorbar(C, ax=ax, fraction=0.02, pad=0.1, label=label)
    plt.show()
    exit(0)

def plot_stress_strain(ax,stress,strain,labels=False,lw=5,ls="-",col="k",ylim=[0,500],xlim=[1.0e-7,0.5]):
    ax.plot(strain, stress,col,ms=1,linestyle=ls,linewidth=lw)
    stress = "$\sigma_{yy}$"
    strain='$'
    strain+="\\varepsilon_{yy}"
    strain+='$'

    x_label = f'Strain ZZ (\%)'
    y_label = f'Stress ZZ (MPa)'
    if labels:
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label,labelpad=10)
    # Compile labels for the graphs
    plt.ylim(ylim)
    plt.xlim(xlim)
    #ax.legend()

def individual_svs(path,sim,show_yield=False,outpath= "",show_offset=False,show=False):
    simulation = fepx_sim(sim,path=path+"/"+sim)
    # Check if simulation value is available 
    outpath =path+"/imgs/" if outpath== "" else outpath
    # if not post process and get
    try:
        stress=simulation.get_output("stress",step="all",comp=2)
        strain=simulation.get_output("strain",step="all",comp=2)
    except:
        simulation.post_process(options="-resmesh stress,strain")
        stress=simulation.get_output("stress",step="all",comp=2)
        strain=simulation.get_output("strain",step="all",comp=2)
    #
    # calculate the yield values
    yield_values = find_yield(stress,strain)
    ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
    stress_off = yield_values["stress_offset"]
    #
    fig, ax = plt.subplots(1, 1)
    plot_stress_strain(ax,stress,strain,lw=1,ylim=[0,1.1*max(stress)],xlim=[1.0e-7,max(strain)+0.001])
    
    if show_yield: ax.plot(ystrain,ystress,"k*",ms=20,label="$\sigma_y$="+str(yield_values["y_stress"]))
    # if show_offset: ax.plot(strain,stress_off,"ko--",ms=5)
    
    fig.subplots_adjust(left=0.15, right=0.97,top=0.98,  bottom=0.11, wspace=0.1, hspace=0.1)        
    
    if show:
        plt.show()
    else:
        fig.savefig(sim+"_svs.png")
#
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
    
def plot_node(ax,coo,vel="",disp="",cc="k"):
    ax.scatter(coo[0],coo[1],coo[2],color=cc)
    if vel!="":
        ax.quiver(coo[0],coo[1],coo[2],vel[0],vel[1],vel[2],color="r")
    if disp!="":
        ax.quiver(coo[0]-disp[0],coo[1]-disp[1],coo[2]-disp[2],disp[0],disp[1],disp[2],color="b")

def plot_multi_nodes(ax,coo,ids,vel="",disp=""):
    for id_val in ids:
        if vel !=disp !="":
            plot_node(ax,coo[id_val],vel[id_val],disp[id_val])
        else:
            plot_node(ax,coo[id_val],cc="y")

def plot_rod_outline(ax):
    #
    a = [ (2**0.5)-1,   3-(2*(2**0.5)),  ((2**0.5)-1)]
    b = [ (2**0.5)-1,     ((2**0.5)-1), 3-(2*(2**0.5))]
    c = [ 3-(2*(2**0.5)),   (2**0.5)-1,   ((2**0.5)-1)]
    #
    X= [a[0],b[0],c[0]]
    Y= [a[1],b[1],c[1]]
    Z= [a[2],b[2],c[2]]

    #print("x",X)
    #print("y",Y)
    #print("z",Z)

    neg_x= [-i for i in X]
    neg_y= [-i for i in Y]
    neg_z= [-i for i in Z]

    X+=neg_x+X    +X    +neg_x+X    +neg_x+neg_x
    Y+=neg_y+Y    +neg_y+Y    +neg_y+neg_y+Y
    Z+=neg_z+neg_z+Z    +Z    +neg_z+Z    +neg_z


    # Plot the 3D surface
    ax.scatter(X,Y,Z,marker=".",c="k")
    #ax.plot_trisurf(X, Y, Z,alpha=0.3)
    ax.scatter(0, 0, 0,color="k")
    #https://stackoverflow.com/questions/67410270/how-to-draw-a-flat-3d-rectangle-in-matplotlib
    for i in range(0,len(X),3):
            verts = [list(zip(X[i:i+3],Y[i:i+3],Z[i:i+3]))]
            ax.add_collection3d(Poly3DCollection(verts,color="k",alpha=0.001))

    s = 3-(2*(2**0.5))
    l = (2**0.5)-1
    new_X= [l, l,  l,  l,  l,  l,  l, l]
    new_Y= [l, s, -s, -l, -l, -s,  s, l]
    new_Z= [s, l,  l,  s, -s, -l, -l, -s]
    vals = [new_X,new_Y,new_Z]
    for i in range(3):
            verts = [list(zip(vals[i-2],vals[i-1],vals[i]))]
            ax.add_collection3d(Poly3DCollection(verts,color="k",alpha=0.001))
    #
    #
    new_X= [-l, -l,  -l,  -l,  -l,  -l,  -l, -l]
    vals = [new_X,new_Y,new_Z]
    for i in range(3):
            verts = [list(zip(vals[i-2],vals[i-1],vals[i]))]
            ax.add_collection3d(Poly3DCollection(verts,color="grey",alpha=0.001))
    #
#
# if __name__=="__main__":
#     sim_name = "homogenous_rcl0_235.sim" #"01_bcc_reference.sim"
#     main_path = "/home/etmengiste/jobs/SERDP/dense_mesh/"#code/FEPX-dev/build/"
#     simulation = fepx_sim(sim_name,main_path+sim_name)
#     step_num = 13
#     node_origin = simulation.get_output("coo",ids="all",step=0)
#     #
#     node_coo = simulation.get_output("coo",ids="all",step=step_num)
#     node_disp = simulation.get_output("disp",ids="all",step=step_num)
#     node_vel = simulation.get_output("vel",ids="all",step=step_num)
#     #
#     id_vals = [i for i in range(len(node_coo))]#100)] #[10,1]#
#     # #
#     print(len(id_vals[::4000]))
#     # exit(0)
#     fig = plt.figure(figsize=(13.5,23))
#     ax = fig.add_subplot(projection='3d')
#     plot_multi_nodes(ax,node_origin,id_vals[::4000])
#     plot_multi_nodes(ax,node_coo,id_vals[::4000],node_disp,node_vel)
#     #figure out element connectivity internally
#     plt.show()
#     # print(simulation.get_summary())