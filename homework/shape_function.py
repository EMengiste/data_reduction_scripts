import os
from itertools import chain, combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import numpy as np
from scipy.linalg import polar
import math

def coordinate_axis(ax,ori,leng = 0.2,offset_text=1.6,col="k",
            lw= 1, offset= np.array([0.01,0,-0.0001]), 
            axis_basis = [[1,0,0],[0,1,0],[0,0,1]],
            xyz_offset = [[-0.0005,-0.0007,0],[0,-0.001,0],[0,-0.001,-0.0004]],
            sty = "solid",space="",coo_labs=["x","y","z"], fs=5):
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
            #print(axis_basis)
            ax.quiver(start[0],start[1],start[2],
                    eig[0],eig[1],eig[2]
                    ,length=leng,normalize=True
                    ,color="k", linestyle = sty,linewidth=lw)
            #
            leng_text=offset_text*leng
            val_txt=(np.array(eig)*(leng_text))+np.array(start)+np.array(xyz_offset[ind])
            ax.text(val_txt[0],val_txt[1],val_txt[2], rod_labs[ind],fontsize=fs, ha='center',va='center',color=col)
            
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

def jacks(ax,start,leng=[1,1,1],sc=0.2,basis=[[1,0,0],[0,1,0],[0,0,1]],col="k"):
    for ind,eig in enumerate(basis):
        print(sc*leng[ind])
        ax.quiver(start[0],start[1],start[2],
                eig[0],eig[1],eig[2]
                ,length=sc*leng[ind]
                ,color=col)
        ax.quiver(start[0],start[1],start[2],
                -eig[0],-eig[1],-eig[2]
                ,length=sc*leng[ind]
                ,color=col)
        
def vect_to_azim_elev(vect):
    x,y,z = vect
    #mag_tot = (x**2 +y**2 +z**2)**0.5
    mag_xy = (x**2 +y**2)**0.5
    azi = math.degrees(math.asin(y/mag_xy))
    ele = math.degrees(math.atan(z/mag_xy))
    return [ele,azi]

def shape_quad_8(input,lengths=[1,1,1]):
    #isbn 978-93-90385-27-0
    # eqn 13.5.2
    # input xi,eta,zeta
    # lengths l1,l2,l3
    val = ""
    xi,eta,zeta = np.array(input)
    l1,l2,l3 = lengths
    #phi_i = c[0] + c[1]*eta + c[2]*eta + c[3]*zeta + c[4]*xi*eta + c[5]*xi*zeta + c[6]*eta*zeta+c[7]*xi*eta*zeta
    Nodal_shape = [(1-xi/l1)*(1-eta/l2)*(1-zeta/l3), #1
                    (  xi/l1)*(1-eta/l2)*(1-zeta/l3), #2
                    (1-xi/l1)*(  eta/l2)*(1-zeta/l3), #3
                    (  xi/l1)*(  eta/l2)*(1-zeta/l3), #4
                    (1-xi/l1)*(1-eta/l2)*(  zeta/l3), #5
                    (  xi/l1)*(1-eta/l2)*(  zeta/l3), #6
                    (1-xi/l1)*(  eta/l2)*(  zeta/l3), #7
                    (  xi/l1)*(  eta/l2)*(  zeta/l3)] #8
    print("N = ",Nodal_shape,"")
    return Nodal_shape
#
def shape_diff(input,xyz,lengths=[1,1,1]):
    #isbn 978-93-90385-27-0
    # eqn 13.5.2
    # input xi,eta,zeta
    # lengths l1,l2,l3
    xi,eta,zeta = np.array(input)
    l1,l2,l3 = lengths
    dN_xi   = np.array([(-1/l1)*(1-eta/l2)*(1-zeta/l3), #1
                        ( 1/l1)*(1-eta/l2)*(1-zeta/l3), #2
                        (-1/l1)*(  eta/l2)*(1-zeta/l3), #3
                        ( 1/l1)*(  eta/l2)*(1-zeta/l3), #4
                        (-1/l1)*(1-eta/l2)*(  zeta/l3), #5
                        ( 1/l1)*(1-eta/l2)*(  zeta/l3), #6
                        (-1/l1)*(  eta/l2)*(  zeta/l3), #7
                        ( 1/l1)*(  eta/l2)*(  zeta/l3)])#8
    
    dN_eta  = np.array([(1-xi/l1)*(-1/l2)*(1-zeta/l3), #1
                        (  xi/l1)*(-1/l2)*(1-zeta/l3), #2
                        (1-xi/l1)*( 1/l2)*(1-zeta/l3), #3
                        (  xi/l1)*( 1/l2)*(1-zeta/l3), #4
                        (1-xi/l1)*(-1/l2)*(  zeta/l3), #5
                        (  xi/l1)*(-1/l2)*(  zeta/l3), #6
                        (1-xi/l1)*( 1/l2)*(  zeta/l3), #7
                        (  xi/l1)*( 1/l2)*(  zeta/l3)])#8
    
    dN_zeta = np.array([(1-xi/l1)*(1-eta/l2)*(-1/l3), #1
                        (  xi/l1)*(1-eta/l2)*(-1/l3), #2
                        (1-xi/l1)*(  eta/l2)*(-1/l3), #3
                        (  xi/l1)*(  eta/l2)*(-1/l3), #4
                        (1-xi/l1)*(1-eta/l2)*( 1/l3), #5
                        (  xi/l1)*(1-eta/l2)*( 1/l3), #6
                        (1-xi/l1)*(  eta/l2)*( 1/l3), #7
                        (  xi/l1)*(  eta/l2)*( 1/l3)])#8
    #
    print(dN_xi)
    print(dN_eta)
    print(dN_zeta)
    #print(len(dN_zeta.T),len(xyz[0]),l1)
    F11= np.matmul(dN_xi,xyz[0])
    F12= np.matmul(dN_xi,xyz[1])
    F13= np.matmul(dN_xi,xyz[2])

    F21= np.matmul(dN_eta,xyz[0])
    F22= np.matmul(dN_eta,xyz[1])
    F23= np.matmul(dN_eta,xyz[2])

    F31= np.matmul(dN_zeta,xyz[0])
    F32= np.matmul(dN_zeta,xyz[1])
    F33= np.matmul(dN_zeta,xyz[2])

    F = np.array([[F11,F12,F13],
                  [F21,F22,F23],
                  [F31,F32,F33]])
    return F 
#
def mapping(fin_shape,poi):
    x_fin,y_fin,z_fin=fin_shape
    shape = shape_quad_8(poi)
    print("x=",np.dot(np.array(x_fin),shape))
    print("y=",np.dot(np.array(y_fin),shape))
    print("z=",np.dot(np.array(z_fin),shape))
    v1 = np.dot(np.array(x_fin),shape)
    v2 = np.dot(np.array(y_fin),shape)
    v3 = np.dot(np.array(z_fin),shape)
    return v1,v2,v3

def print_latex(matrix,val):
    preamble= val+"&=\\begin{bmatrix}"
    print(preamble)

    for val in matrix:
        print("  &".join(str(val)[1:-1].split()),"\\\\")
    print("\end{bmatrix}\\\\")
   

def draw_cube(ax,X,Y,Z,col="r",alp=0.1):
    # assume kasmer numbering
    top = [4,5,7,6]
    bottom = [0,1,3,2]
    left = [3,7,6,2]
    right = [0,1,5,4]
    front = [1,3,7,5]
    back = [0,2,6,4]
    for side in [top,bottom,left,right,front,back]:
        X_vals = [X[i] for i in side]
        Y_vals = [Y[i] for i in side]
        Z_vals = [Z[i] for i in side]
        verts = [list(zip(X_vals,Y_vals,Z_vals))]
        ax.add_collection3d(Poly3DCollection(verts,color=col,alpha=alp))
#
#
# Initial node coordinates in order
X= [0,1,0,1,0,1,0,1]
Y= [0,0,1,1,0,0,1,1]
Z= [0,0,0,0,1,1,1,1]
#
pt1 = np.array( [0.25,0.25,0.25])
pt2 = np.array([0.75,0.75,0.75])
#
point = [pt1,pt2]
order_coos=[0,2,4,1]

destination = "."
show = True
show = False
#

#
# import text file names for later use
dirs =[ i for i in os.listdir(".") if i.endswith(".txt")]
dirs.sort()
print(dirs)

for i in range(0,3,1):
    print(dirs[i])
    fig,axs = plt.subplots(2,2,figsize=(10,10),subplot_kw=dict(projection='3d'))
    axs = axs.flatten()
    print(axs)
    #exit(0)
    #
    ini_col = "r"
    fin_col = "b"
    ele,azi = vect_to_azim_elev([3,1,1])
    start = -0.5
    end= 1.5
    #
    #
    x_fin,y_fin,z_fin = np.loadtxt(dirs[i])
    #
    # Point 1
    # plot point of interest 1
    for ind,poi in enumerate(point):
        ax2 = axs[ind]
        ax1 = axs[ind-2]
        ax0 = axs[ind-4]
        #ax.label()
        # plot cube
        #draw_cube(ax,X,Y,Z,col=ini_col)
        draw_cube(ax0,X,Y,Z,col=ini_col)
        #draw_cube(ax1,X,Y,Z,col=ini_col)
        # plot given final coordinates
        #draw_cube(ax,x_fin,y_fin,z_fin,col=fin_col)
        #draw_cube(ax0,x_fin,y_fin,z_fin,col=fin_col)
        draw_cube(ax1,x_fin,y_fin,z_fin,col=fin_col)

        F = shape_diff(poi,[x_fin,y_fin,z_fin]).T

        print(np.matmul(F.T,F))
        print_latex(F,"F")
        print("J&=",np.linalg.det(F),"\\\\")
        #
        # https://docs.scipy.org/doc/scipy/reference/generated/scipy.linalg.polar.html
        #
        R,V = polar(F,side="left")
        #print_latex(R,"R")
        R,U = polar(F,side="right")
        #
        #exit(0)
        #print(np.matmul(R,R.T))
        # R.R.T ==I
        v_calc = np.matmul(np.matmul(R,U),R.T)
        #
        #
        print_latex(V,val="V")
        print_latex(R,"R")
        print_latex(U,"U")
        #coordinate_axis(ax,poi)
        # Plot point of interest
        #ax.plot(poi[0],poi[1],poi[2],ini_col+"o")
        ax0.plot(poi[0],poi[1],poi[2],ini_col+"o")


        v1,v2,v3 = mapping([x_fin,y_fin,z_fin],poi)
        #ax.plot(v1,v2,v3,fin_col+"o")

        eig_vals,eig_vect = np.linalg.eig(V)
        jacks(ax1,[v1,v2,v3],eig_vals,basis=eig_vect,col=fin_col)
        jacks(ax1,poi,col=ini_col)        

        eig_vals,eig_vect = np.linalg.eig(U)
        jacks(ax0,poi,eig_vals,basis=eig_vect,col=fin_col)
        jacks(ax0,poi,col=ini_col)        
        #
        #exit(0)
    #
    for ax in axs:
        #
        ax.set_proj_type("persp")
        ax.set_aspect("equal")
        ax.set_box_aspect(aspect = (2,2,2))
        print("vals are",ele,azi)
        ax.view_init(elev=ele, azim=azi)
        ax.set_xlim([start,end])
        ax.set_ylim([start,end])
        ax.set_zlim([start,end])
    plt.grid(False)

    if show:
        plt.show()
    else:
        #fig.savefig("funda_region")
        fig.savefig(str(i)+"box.png",dpi=100)
        #fig.savefig("funda_region_zoomed_grain_id_"+str(grain_id))


exit(0)

v1,v2,v3 = mapping([x_fin,y_fin,z_fin],poi)
ax.plot(v1,v2,v3,fin_col+"o")
coordinate_axis(ax0,[v1,v2,v3],axis_basis=V,col="b")
coordinate_axis(ax0,poi)        
coordinate_axis(ax2,[v1,v2,v3],axis_basis=U,col="b")

x_f,y_f,z_f= np.matmul(U, np.array([X,Y,Z]))
draw_cube(ax1,x_f,y_f,z_f,col=ini_col)
x_f,y_f,z_f= np.matmul(R, np.array([x_f,y_f,z_f]))
draw_cube(ax2,x_f,y_f,z_f,col=ini_col)


x_f,y_f,z_f= np.matmul(R, np.array([X,Y,Z]))
x_f,y_f,z_f= np.matmul(V, np.array([x_f,y_f,z_f]))
draw_cube(ax0,x_f,y_f,z_f,col=ini_col)
coordinate_axis(ax1,[v1,v2,v3],axis_basis=R,col="r")
coordinate_axis(ax1,poi)
coordinate_axis(ax2,[v1,v2,v3],axis_basis=R,col="r")
print("\n\n\n")