import os
from itertools import chain, combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import numpy as np
import math


def coordinate_axis(ax,ori,leng = 0.002,offset_text=1.6,
            lw= 4, offset= np.array([0.01,0,-0.0001]), axis_basis = [[1,0,0],[0,1,0],[0,0,1]],
                    xyz_offset = [[-0.0005,-0.0007,0],[0,-0.001,0],[0,-0.001,-0.0004]],
                    sty = "solid",space="rod_real", fs=60):
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

def angle_axis_to_mat(angle,axis,scale=1):
    cos_thet = math.cos(angle)
    sin_thet = math.sin(angle)
    u_x,u_y,u_z = axis
    r11 = cos_thet + (u_x ** 2 * (1-cos_thet))
    r12 = (u_x * u_y * (1-cos_thet)) - (u_z * sin_thet)
    r13 = (u_x * u_z * (1-cos_thet)) + (u_y * sin_thet)

    r21 = (u_y * u_x * (1-cos_thet)) + (u_z * sin_thet)
    r22 = cos_thet + (u_y ** 2 * (1-cos_thet))
    r23 = (u_y * u_z * (1-cos_thet)) - (u_x * sin_thet)

    r31 = (u_z * u_x * (1-cos_thet)) - (u_y * sin_thet)
    r32 = (u_z * u_y * (1-cos_thet)) + (u_x * sin_thet)
    r33 = cos_thet + (u_z ** 2 * (1-cos_thet))
    Rot_Mat = [ [r11,r12,r13],
                [r21,r22,r23],
                [r31,r32,r33]]
    return Rot_Mat

def plot_box(matrix,start=[0,0,0],bcc=False,fcc=False,col="b",ms=5):


    if fcc:
        ### Top of X
        X+= [  1,0.5,0.5,0.5]
        Y+= [0.5,0  ,0.5,  1]
        Z+= [0.5,0.5,  0,0.5]

        # Bottom of x
        X+=[0  ,0.5,0.5,0]
        Y+=[0.5,0.5,0,0]
        Z+=[0.5,1  ,0.5,0]
    elif bcc:
        X +=[0.5]
        Y +=[0.5]
        Z +=[0.5]

    vects = [ ]
    for vect in zip(X,Y,Z):
        print(vect)
        print(matrix)
        
        vectors= np.dot(vect,np.array(matrix))+start
        print(vectors)
        vects.append(vectors)
    vects =np.array(vects).T

    dots=ax.plot(X,Y,Z,col+"o",ms=ms)
    j=4
    #https://stackoverflow.com/questions/67410270/how-to-draw-a-flat-3d-rectangle-in-matplotlib
    for i in range(0,len(X),j):
            list1=list(zip(list(X[i:i+j]),list(Y[i:i+j]),list(Z[i:i+j])))
            print(list)
            verts = [list1]
            boxes = ax.add_collection3d(Poly3DCollection(verts,color=col,linewidth=3,alpha=0.1))

    
    return [dots,boxes]

def plot_crystal(ax,start=[0,0,0],type="",lw=5,matrix="",c="k",c2="r",ec="k",ms=20):
    X=[1,0,0,0,0,1,1,1,1]
    Y=[1,1,1,0,0,1,0,0,1]
    Z=[1,0,1,1,0,1,1,0,0]
    if matrix=="":
        matrix = [[1,0,0],[0,1,0],[0,0,1]]
    if type == "bcc":
        X+=[1/2]
        Y+=[1/2]
        Z+=[1/2]
    if type == "fcc":
        X+=[  1,1/2,1/2,1/2,1/2,  0]
        Y+=[1/2,  1,1/2,1/2,  0,1/2]
        Z+=[1/2,1/2,  1,  0,1/2,1/2]

    vects = [np.dot(np.array([x,y,z]),np.array(matrix))+start
                        for x,y,z in zip(X,Y,Z) ]
    X,Y,Z= np.array(vects).T
    ax.scatter(X[:9],Y[:9],Z[:9],c=c,s=ms,ec=ec,lw=lw)
    ax.scatter(X[9:],Y[9:],Z[9:],c=c2,s=ms,ec=ec,lw=lw)


def vect_to_azim_elev(vect):
    x,y,z = vect
    mag_tot = (x**2 +y**2 +z**2)**0.5
    mag_xy = (x**2 +y**2)**0.5
    azi = math.degrees(math.asin(y/mag_xy))
    ele = math.degrees(math.atan(z/mag_xy))
    return [ele,azi]


def draw_lattice(type="bcc",scale=20,dims=[1,1,1],axis = [1,0,0],lw=5,ec="k",ms=50,angle = 0,offset=np.array([0,0,0])):
    for i in range(0,dims[0]):
        for j in range(0,dims[1]):
            for k in range(0,dims[2]):
                #scale up square
                matrix = np.array(angle_axis_to_mat(angle,axis))*scale
                start = np.dot(np.array([i,j,k])+offset,matrix)
                plot_crystal(ax,start=start,type=type,lw=lw,ec=ec,matrix=matrix,ms=ms)

def update(frame):
    # for each frame, update the data stored on each artist.
    x = X1[:frame]
    y = Y1[:frame]
    z = Z1[:frame]
    # update the scatter plot:
    data = np.stack([x, y, z]).T
    scat.set_offsets(data)
    # update the line plot:
    line2.set_xdata(X1[:frame])
    line2.set_ydata(Y1[:frame])
    line2.set_zdata(Z1[:frame])
    return (scat, line2)

# Latex interpretation for plots
plt.rcParams.update({'font.size': 15})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'Dejvu Sans'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.figsize'] = 8,8

destination = "/home/etmengiste/documents/proposal_slides/slides_proposal/figures/"
fig = plt.figure()
ax = fig.add_subplot(projection='3d')

scale=60
ms=200
width=50
lw=3
axis = [1,0,0]
angle = 0
ec="k"
type="bcc"
plot_crystals=False
spacing=scale+width
offset=np.array([2.3,0,0])

dims = [1,1,1]
draw_lattice(type="",scale=20,dims=dims,axis = axis,lw=lw,ec=ec,ms=ms,angle =angle,offset=offset)

offset=np.array([0,0,0])

dims = [1,1,1]
draw_lattice(type=type,scale=20,dims=dims,axis = axis,lw=lw,ec=ec,ms=ms,angle =angle,offset=offset)

offset=np.array([-2.3,0,0])

dims = [1,1,1]
draw_lattice(type="fcc",scale=20,dims=dims,axis = axis,lw=lw,ec=ec,ms=ms,angle =angle,offset=offset)


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
plt.grid(False)
plt.tight_layout()

show = True
show = False

if show:
       plt.show()
else:
       #fig.savefig("funda_region")
       fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
       bbox = fig.bbox_inches.from_bounds(1.7,2,5, 4)
       fig.savefig(destination+"crystal_types.png", bbox_inches=bbox)
       #fig.savefig("funda_region_zoomed_grain_id_"+str(grain_id))

exit(0)

dims = [1,3,3]
draw_lattice(type=type,scale=20,dims=dims,axis = axis,lw=lw,ec=ec,ms=ms,angle =angle,offset=offset)


offset=np.array([0,2,0])
angle = 45
draw_lattice(type=type,scale=20,dims=dims,axis = axis,lw=lw,ec=ec,ms=ms,angle =angle,offset=offset)
offset=np.array([0,-3,-4])
dims = [1,2,4]
angle = 70
draw_lattice(type=type,scale=20,dims=dims,axis = axis,lw=lw,ec=ec,ms=ms,angle =angle,offset=offset)

offset=np.array([0,-2,-4])
dims = [1,2,4]
angle =50
draw_lattice(type=type,scale=20,dims=dims,axis = axis,lw=lw,ec=ec,ms=ms,angle =angle,offset=offset)



spacing+=scale+40
# coordniate system
offset= np.array([0,0,0])
start = np.array([0,-50,-100])
xyz_offset = [[0,0,0],[0,10,0],[0,0,0]]
scale=1
matrix = [[scale,0,0],[0,scale,0],[0,0,scale]]
coordinate_axis(ax,start,space="real",lw=3,axis_basis=matrix,fs=20,leng=20,offset_text=1.5,offset=offset,xyz_offset=xyz_offset)


spacing+=scale+width

view="1"

ele,azi,roll =[-1,180,0]
#ele,azi = vect_to_azim_elev([4,-5,4])
print(ele,azi)
print(spacing)
ax.view_init(elev=ele, azim=azi)
ax.set_xlim([-50,spacing])
ax.set_ylim([-50,60])
#ax.set_zlim([-1,60])
ax.set_aspect("equal")
ax.set_proj_type("ortho")
ax.axis("off")
plt.grid(False)
plt.tight_layout()

show = True
show = False

if show:
       plt.show()
else:
       #fig.savefig("funda_region")
       fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
       bbox = fig.bbox_inches.from_bounds(1, 1, 6, 6)
       fig.savefig(destination+"polycrystal_"+type+"_"+view+".png", bbox_inches=bbox)
       #fig.savefig("funda_region_zoomed_grain_id_"+str(grain_id))

exit(0)


### Top of X
X= [1,1,1,1]
Y= [1,0,0,1]
Z= [1,1,0,0]

# Bottom of x
X+=[0,0,0,0]
Y+=[1,1,0,0]
Z+=[0,1,1,0]

#Top of y
X+=[1,0,0,1]
Y+=[0,0,0,0]
Z+=[1,1,0,0]

#Bottom of y
X+=[1,0,0,1]
Y+=[1,1,1,1]
Z+=[1,1,0,0]

#Top of z
X+=[1,0,0,1]
Y+=[1,1,0,0]
Z+=[1,1,1,1]

#Bottom of z
X+=[1,0,0,1]
Y+=[1,1,0,0]
Z+=[0,0,0,0]
i,j =[0,0]


axis = [0,0,1]
angle = 0
shear_component = 0.3

offset= np.array([0,0,0])
start=np.array([0,0,0])
xyz_offset = [[0,0,0],[0,0.32,0],[0,0,0]]

matrix = angle_axis_to_mat(angle,axis)
start = np.array([50,50,0])

coordinate_axis(ax,start,space="real",axis_basis=matrix,fs=20,leng=15,offset_text=1.25,offset=offset,xyz_offset=xyz_offset)

start = np.array([-90,0,0])
scale = 50
#scale up square
matrix = [[scale,0,0],[0,scale,0],[0,0,scale]]

vects = [np.dot(np.array([x,y,z]),np.array(matrix))+start
                    for x,y,z in zip(X,Y,Z) ]
X1,Y1,Z1= np.array(vects).T

plot_box(matrix,start,ms=30,bcc=False,col="k")

ax.scatter(0.5*X1[0],0.5*Y1[0],0.5*Z1[0],c="k",s=30)

start = np.array([0,0,0])
#scale up square
matrix = [[scale,0,0],[0,scale,0],[0,0,scale]]
vects = [np.dot(np.array([x,y,z]),np.array(matrix))+start for x,y,z in zip(X,Y,Z) ]
X1,Y1,Z1= np.array(vects).T
plot_box(X1,Y1,Z1,ms=30,fcc=True,col="k")


text = ax.text(X1[0],Y1[0],Z1[0]," rotation of angle"+str(angle)+" about axis "+str(axis))
for i in range(3):
    for j in range(5):
        dots,boxes = plot_box(X1+i,Y1,Z1+j,col="r")
plt.show()
exit(0)

fig = plt.figure()
ax = fig.add_subplot(projection='3d')


X=[1,1,1,1,0,0,0,0]
Y=[1,0,0,1,1,1,0,0]
Z=[1,1,0,0,0,1,1,0]


offset= np.array([0,0,0])
start=np.array([0,0,0])
xyz_offset = [[0,0,0],[0,0.32,0],[0,0,0]]
coordinate_axis(ax,start,space="real_latex",fs=55,leng=.4,offset_text=1.25,offset=offset,xyz_offset=xyz_offset)
#coordinate_axis(ax,start,space="real_latex",fs=55,leng=0.04,offset_text=1.4,offset=offset,xyz_offset=xyz_offset)
ax.text(X[0],Y[0],Z[0]," Original")
plot_box(X,Y,Z,col="k")


axis = [0,0,1]
angle = 20
shear_component = 0.3


matrix = angle_axis_to_mat(angle,axis)
start = np.array([4,0,0])

vects = [np.dot(np.array([x,y,z]),np.array(matrix))+start for x,y,z in zip(X,Y,Z) ]
X1,Y1,Z1= np.array(vects).T

ax.text(X1[0],Y1[0],Z1[0]," rotation of angle"+str(angle)+" about axis "+str(axis))
plot_box(X1,Y1,Z1,col="r")

start = np.array([2,0,0])

matrix = [[1,0,shear_component],[0,1,0],[0,0,1]]
vects = [np.dot(np.array([x,y,z]),np.array(matrix))+start for x,y,z in zip(X,Y,Z) ]
X2,Y2,Z2 = np.array(vects).T

ax.text(X2[0],Y2[0],Z2[0]," simple shear")
plot_box(X2,Y2,Z2,col="g")
coordinate_axis(ax,start,space="real",axis_basis=matrix,fs=55,leng=.4,offset_text=1.25,offset=offset,xyz_offset=xyz_offset)


start = np.array([6,0,0])

matrix = [[1,0,shear_component],[0,1,0],[shear_component, 0,1]]
vects = [np.dot(np.array([x,y,z]),np.array(matrix))+start for x,y,z in zip(X,Y,Z) ]
X3,Y3,Z3 = np.array(vects).T

ax.text(X3[0],Y3[0],Z3[0]," pure shear")
plot_box(X3,Y3,Z3,col="b")
coordinate_axis(ax,start,space="real",axis_basis=matrix,fs=55,leng=.4,offset_text=1.25,offset=offset,xyz_offset=xyz_offset)



#ax.scatter(4,-5,4,"ko")
#ele,azi,roll =[31,-28,0]
# 28,-58
ele,azi,roll =[35,45,0]
ele,azi = vect_to_azim_elev([4,-5,4])
print(ele,azi)
ax.view_init(elev=ele, azim=azi)
ax.set_xlim([-1,6])
ax.set_ylim([-1,3])
ax.set_zlim([-1,2])
#ax.set_aspect("equal")
ax.set_proj_type("persp")
ax.axis("off")
plt.grid(False)
show = True
#show = False

if show:
       plt.show()
else:
       #fig.savefig("funda_region")
       fig.savefig("coo_ax")
       #fig.savefig("funda_region_zoomed_grain_id_"+str(grain_id))

exit(0)
os.system('convert -gravity south +append coo_ax.png /media/schmid_2tb_1/etmengiste/files/slip_system_study/common_files/Cube_msh.png Cube_msh.png')
os.system("convert -chop 195x0+265+0 Cube_msh.png Cube_msh.png")
exit(0)
os.system("convert coo_ax.png -crop 380x400+190+300 coo_ax.png")
os.system("convert -gravity south +append coo_ax.png grain.png combined_grain_coo_ax.png")
os.system("convert combined_grain_coo_ax.png -chop 180x0+380+0 axis_grain.png")
exit(0)
#plot_rod_outline(ax)
start= [0,0,0]
show_axies = True
debug = True
debug = False
rod_labs = ["$r_1 , x$","$r_2 , y$","$r_3, z$"]
xyz_labs = ["$x$","$y$","$z$"]
axis_basis = np.array([[1,0,0],[0,1,0],[0,0,1]])
offset= np.array([1.5,-1.5,-1.5])
start = np.array(start)+offset
fs=0.6
sty = "solid"
val = 1
leng = 0.5
lab_offset = np.array([0.0004,0.0004,0.0005])
lw= 4
##
##     make into function
##

leng*=val
if show_axies:
        for ind,eig in enumerate(axis_basis):
                ax.quiver(start[0],start[1],start[2],
                    eig[0],eig[1],eig[2]
                    ,length=leng,normalize=True
                    ,color="k", linestyle = sty,linewidth=lw)
                #
                val_txt=(np.array(eig)*leng)+np.array(start)
                ax.text(val_txt[0],val_txt[1],val_txt[2], rod_labs[ind],fontsize=fs, color='k')
                
                if debug:
                    start = np.array([0,0,0])
                    leng = 0.6
                    lab_offset = np.array([0.0005,0.00,0.0007])
                    lw= 5.6
                    val_txt=(np.array(eig)*leng)+np.array(start)
                    ax.text(val_txt[0],val_txt[1],val_txt[2], rod_labs[ind],fontsize=fs, color='k')
                    ax.quiver(start[0],start[1],start[2],
                            eig[0],eig[1],eig[2]
                            ,length=leng,normalize=True
                            ,color="k", linestyle = sty,linewidth=lw)
                    #
                #
                #
####
#just playing
ax.plot(0,0,0,'ko')
scale = 0.9
axis_basis =[[ 1, 1, 1],
             [-1,-1,-1],
             [ 1, 1,-1],
             [ 1,-1, 1],
             [ 1, 1,-1],
             [ 1, -1,-1],
             [-1,-1,-1]]
axis_basis+=[[-1,-1,-1],
             [ 1, 1, 1],
             [-1,-1, 1],
             [-1, 1,-1],
             [-1,-1, 1],
             [-1, 1, 1],
             [ 1, 1, 1]]
ms=0.5
axis_basis =scale*np.array(axis_basis)
for i in range(0,len(axis_basis),2):
       eig= axis_basis[i]
       eig1= eig
       ax.plot(eig1[0],eig1[1],eig1[2],"ko",ms=20*ms)
       eig= axis_basis[i-1]
       eig2= eig
       ax.plot(eig2[0],eig2[1],eig2[2],"ko",ms=20*ms)
       x,y,z = [[eig1[0],eig2[0]],
                [eig1[1],eig2[1]],
                [eig1[2],eig2[2]]]
       ax.plot(x,y,z,"r-")


#https://stackoverflow.com/questions/464864/get-all-possible-2n-combinations-of-a-list-s-elements-of-any-length
def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)  # allows duplicate elements
    return chain.from_iterable(combinations(s, r) for r in range(len(s)+1))

lVals = [i for i in range(len(axis_basis))]
lVals = [(i) for i in powerset(lVals) if len(i)==4]
#print(lVals)
for val in lVals:
    #print(verts)
    #print(axis_basis[val[0]],axis_basis[val[1]],axis_basis[val[2]],axis_basis[val[2]])
    values = [axis_basis[i] for i in val]
    #print(list(values))
    #exit(0)
    verts = values
    #print(np.shape(verts))
    #ax.add_collection3d(Poly3DCollection(verts,color="r",alpha=1))
    ####

#exit(0)

#ax.set_aspect("equal")
#ax.axis("off")
plt.grid(False)

ele,azi,roll =[35,45,0]
#ele,azi,roll =[26,62,0]
ax.view_init(elev=ele, azim=azi)

show = True
#show = False
if show:
       plt.show()

exit(0)
'''
else:
    #fig.savefig("funda_region")
    fig.savefig("funda_region_zoomed")
    #fig.savefig("funda_region_zoomed_grain_id_"+str(grain_id))'''