import os
from ezmethods import *
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
import math
# Latex interpretation for plots
# Latex interpretation for plots
plt.rcParams.update({'font.size': 55})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.figsize'] = 20,20

# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1
pi = math.pi
def Cubic_sym_quats():
       AngleAxis =  np.array([[0.0     , 1 ,   1,    1 ],   # % identity
                     [pi*0.5  , 1 ,   0,    0 ],   # % fourfold about x1
                     [pi      , 1 ,   0,    0 ],   #
                     [pi*1.5  , 1 ,   0,    0 ],   #
                     [pi*0.5  , 0 ,   1,    0 ],   # % fourfold about x2
                     [pi      , 0 ,   1,    0 ],   #
                     [pi*1.5  , 0 ,   1,    0 ],   #
                     [pi*0.5  , 0 ,   0,    1 ],   # % fourfold about x3
                     [pi      , 0 ,   0,    1 ],   #
                     [pi*1.5  , 0 ,   0,    1 ],   #
                     [pi*2/3  , 1 ,   1,    1 ],   # % threefold about 111
                     [pi*4/3  , 1 ,   1,    1 ],   #
                     [pi*2/3  ,-1 ,   1,    1 ],   # % threefold about 111
                     [pi*4/3  ,-1 ,   1,    1 ],   #
                     [pi*2/3  , 1 ,  -1,    1 ],   # % threefold about 111
                     [pi*4/3  , 1 ,  -1,    1 ],   #
                     [pi*2/3  ,-1 ,  -1,    1 ],   # % threefold about 111
                     [pi*4/3  ,-1 ,  -1,    1 ],   #
                     [pi      , 1 ,   1,    0 ],   # % twofold about 110
                     [pi      ,-1 ,   1,    0 ],   #
                     [pi      , 1 ,   0,    1 ],   #
                     [pi      , 1 ,   0,   -1 ],   #
                     [pi      , 0 ,   1,    1 ],   #
                     [pi      , 0 ,   1,   -1 ]])

       cubic_sym = np.array([quat_of_angle_ax(a[0],a[1:]) for a in AngleAxis])
       return cubic_sym


def quat_prod(q1, q2,debug=False):
       # input a q1 and 4*1
       many =False
       try:
              a = np.array([q[0] for q in q1.T])
              b = np.array([q[0] for q in q2.T])
              avect= np.array([q[1:] for q in q1.T])
              bvect= np.array([q[1:] for q in q2.T])
              if debug:
                     print("a",avect.shape)
                     print("b",bvect.shape)
              many=True
       except:
              a=q1[0]
              b=q2[0]
              avect=q1[1:]
              bvect=q2[1:]
       #
       a3 = np.tile(a,[3,1]).T
       b3 = np.tile(b,[3,1]).T

       if debug:
              print("a3",a3.shape)
       #
       if many:
              dotted_val = np.array([np.dot(a,b) for a,b in zip(avect,bvect)])
              crossed_val = np.array([np.cross(a,b) for a,b in zip(avect,bvect)])
       else:
              dotted_val=np.dot(avect,bvect)
              crossed_val=np.cross(avect, bvect)
       
       #
       ind1 =a*b - dotted_val
       if debug:
              print(ind1.shape)

       #print(crossed_val.shape)
       #print((b3*avect).shape)
       val=a3*bvect + b3*avect +crossed_val

       if debug:
              print(val)
       quat = np.array([[i,v[0],v[1],v[2]] for i,v in zip(ind1, val)])
       #
       if many:
              max = 0
              max_ind=0
              for ind,q in enumerate(quat):
                     if q[0]<0:
                            quat[ind] = -1*quat[ind]
                     if ind==0:
                            max= quat[ind][0]
                            max_ind=ind
                     elif quat[ind][0]>max:
                            #print("larger")
                            print(quat[ind])
                            max=quat[ind][0]
                            max_ind=ind
                     #
              value = normalize_vector(quat[max_ind])
              #print("--------",value)
              return quat[max_ind]
       else:
              if quat[0]<0:
                     quat=-1*quat
              #print(quat)
              return quat



def quat_of_angle_ax(angle, raxis):
       #
       half_angle = 0.5*angle
       #
       cos_phi_by2 = math.cos(half_angle)
       sin_phi_by2 = math.sin(half_angle)
       #
       rescale = sin_phi_by2 / np.sqrt(np.dot(raxis,raxis))
       quat = np.append([cos_phi_by2],np.tile(rescale,[3])*raxis)
       if cos_phi_by2<0:
              quat = -1*quat
       #
       #
       return quat

def ret_to_funda(quat, sym_operators,debug=False):
       m = len(sym_operators)
       n = 1
       # if passing a set of symetry operators make sure to [quat]
       tiled_quat = np.tile(quat,(1,m))
       if debug:
              print("\n\n\n+++++ ret to funda ++++")
              pprint(tiled_quat,preamble="tiled")
              print(tiled_quat.shape)
       #
       reshaped_quat=tiled_quat.reshape(4,m*n,order='F').copy()
       sym_operators=sym_operators.T
       if debug:
              pprint(reshaped_quat,preamble="reshaped")
              print("old shape",sym_operators.shape)
              pprint(sym_operators)
       if debug:
              print(sym_operators.shape)
       equiv_quats = quat_prod(reshaped_quat,np.tile(sym_operators,(1,n)))
       #pprint(equiv_quats)
       #exit(0)
       return equiv_quats



def rod_to_quat(val,debug=False):
       norm,mag = normalize_vector(val,magnitude=True)
       omega= 2*math.atan(mag)
       if debug:
              print("an ax",quat_of_angle_ax(omega,norm))
              print(omega)
       s= math.sin(omega/2)
       c= math.cos(omega/2)
       values = np.array([c, s*norm[0],s*norm[1], s*norm[2]])
       return values

def quat_to_rod(val):
       #print(val)
       omega = 2*math.acos(val[0])
       n = val[1:]/math.sin(omega/2)
       r = math.tan(omega/2)*n
       return r



ax = plt.figure().add_subplot(projection='3d')
b = [ (2**0.5)-1,   3-(2*(2**0.5)),  ((2**0.5)-1)]
c = [ (2**0.5)-1,     ((2**0.5)-1), 3-(2*(2**0.5))]
d = [ 3-(2*(2**0.5)),   (2**0.5)-1,   ((2**0.5)-1)]

X= [b[0],c[0],d[0]]
Y= [b[1],c[1],d[1]]
Z= [b[2],c[2],d[2]]

print("x",X)
print("y",Y)
print("z",Z)

neg_x= [-i for i in X]
neg_y= [-i for i in Y]
neg_z= [-i for i in Z]

X+=neg_x+X    +X    +neg_x+X    +neg_x+neg_x
Y+=neg_y+Y    +neg_y+Y    +neg_y+neg_y+Y
Z+=neg_z+neg_z+Z    +Z    +neg_z+Z    +neg_z


# Plot the 3D surface
ax.scatter(X,Y,Z,marker=".",c="k",s=50)
#ax.plot_trisurf(X, Y, Z,alpha=0.3)
ax.scatter(0, 0, 0,color="k")
ax.set(xlim=(-2.5, 2.5), ylim=(-2.5, 2.5), zlim=(-2.5, 2.5),
       xlabel='X', ylabel='Y', zlabel='Z')

ax.set(xlim=(-.6, .6), ylim=(-.6, .6), zlim=(-.6, .6),
       xlabel='X', ylabel='Y', zlabel='Z')
ax.set_aspect("equal")
#https://stackoverflow.com/questions/67410270/how-to-draw-a-flat-3d-rectangle-in-matplotlib
for i in range(0,len(X),3):
    verts = [list(zip(X[i:i+3],Y[i:i+3],Z[i:i+3]))]
    ax.add_collection3d(Poly3DCollection(verts,color="grey",alpha=0.1))

s = 3-(2*(2**0.5))
l = (2**0.5)-1
new_X= [l, l,  l,  l,  l,  l,  l, l]
new_Y= [l, s, -s, -l, -l, -s,  s, l]
new_Z= [s, l,  l,  s, -s, -l, -l, -s]
vals = [new_X,new_Y,new_Z]
for i in range(3):
       verts = [list(zip(vals[i-2],vals[i-1],vals[i]))]
       ax.add_collection3d(Poly3DCollection(verts,color="grey",alpha=0.1))

new_X= [-l, -l,  -l,  -l,  -l,  -l,  -l, -l]
vals = [new_X,new_Y,new_Z]
for i in range(3):
       verts = [list(zip(vals[i-2],vals[i-1],vals[i]))]
       ax.add_collection3d(Poly3DCollection(verts,color="grey",alpha=0.1))

def make_quat(arr):
       vals= []
       for i in arr:
              vals.append(rod_to_quat(i))
       
       return np.array(vals).T
cub = Cubic_sym_quats()




simulations = os.listdir(home)
simulations.sort()
simulations.remove("common_files")
home="/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"
colors = ["k", (10/255,10/255,10/255),
          (70/255,70/255,70/255),
          (150/255,150/255,150/255),
          (200/255,200/255,200/255),
          (230/255,230/255,230/255)]
for val,col in zip([75,0,1,2,3,4],colors):
       print(simulations[val])
       sim = fepx_sim("Cube.sim",path=home+simulations[val]+"/Cube.sim")
       step= "28"
       for id in range(500,600):
              stress = sim.get_output("stress",step=step,res="elsets",ids=[id])
              ori = sim.get_output("ori",step=step,res="elsets",ids=[id])
              print("rod ini",ori)
              ori= rod_to_quat(ori)
              print("quat ini",ori)
              ori = ret_to_funda(ori,cub)
              print("quat fin",ori)

              ori= quat_to_rod(ori)
              print("rod fin",ori)
              #exit(0)
              stress_mat= to_matrix(stress)
              eig_val, eig_vect = np.linalg.eig(stress_mat)
              print("eig_val",eig_val)
              eig_val = normalize_vector(eig_val)
              print("eig_val",eig_val)
              pprint(eig_vect)
              ax.scatter(ori[0],ori[1],ori[2])
              start = ori#[0, 0 ,0]
              for ind,eig in enumerate(eig_vect):
                     ax.quiver(start[0],start[1],start[2],
                            start[0]+eig[0],start[1]+eig[1],start[2]+eig[2]
                            ,length=0.01,normalize=True,color=col)


ax.view_init(elev=35., azim=45, roll=0)
plt.savefig("funda_region")
exit(0)

# https://www.geeksforgeeks.org/python-get-the-indices-of-all-occurrences-of-an-element-in-a-list/#
surf1= [ind for ind, elt in enumerate(X) if  elt ==  b[0]]
Vx= [X[i]for i in surf1]
Vy= [Y[i]for i in surf1]
Vz= [Z[i]for i in surf1]
verts = [list(zip(Vx,Vy,Vz))]
#ax.add_collection3d(Poly3DCollection(verts,zsort=min,alpha=0.1))
surf2= [ind for ind, elt in enumerate(X) if  elt ==  -b[0]]

surf3= [ind for ind, elt in enumerate(Y) if  elt ==  b[0]]
surf4= [ind for ind, elt in enumerate(Y) if  elt ==  -b[0]]

surf5= [ind for ind, elt in enumerate(Z) if  elt ==  b[0]]
surf6= [ind for ind, elt in enumerate(Z) if  elt ==  -b[0]]
 
# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1
    

ax = plt.figure().add_subplot(projection='3d')
X= [20.0, 0.0, 0.0, 0.0, 10.0, 20.0, 30.0, 10.0, 30.0, 10.0, 20.0, 40.0, 30.0, 40.0, 40.0, 25.0, 30.0, 35.0, 25.0, 35.0, 25.0, 30.0, 60.0, 35.0, 20.0, 40.0, 60.0, 20.0, 40.0, 60.0]
Y= [20.0, 20.0, 10.5, 1.0, 15.25, 15.25, 15.25, 10.5, 10.5, 5.75, 5.75, 20.0, 5.75, 15.25, 5.75, 12.875, 12.875, 12.875, 10.5, 10.5, 8.125, 8.125, 20.0, 8.125, 10.5, 10.5, 10.5, 1.0, 1.0, 1.0]
Z= [22.76535053029903, 38.96497960559887, 66.77662089771968, 95.93421141486863, 28.38618575891917, 15.135746066298575, 18.89683134970845, 42.13047903563476, 12.313994721968864, 56.269231987969555, 32.07046324924786, 55.2219601120633, 12.35786940187457, 41.921945008763586, 28.16061601330233, 10.070378551835649, 13.476236202374565, 23.735327551549496, 11.389019427629599, 21.305869908317952, 13.20425316321665, 11.845200358118927, 100.34872346753596, 20.213927716104187, 18.26818405993052, 30.495051027879345, 74.47058506243454, 54.23429875499636, 24.48021292531605, 29.959116124492354]

# Plot the 3D surface
ax.scatter(X,Y,Z,marker=".")
ax.plot_trisurf(X, Y, Z,alpha=0.3)

plt.show()

exit(0)
# Plot projections of the contours for each dimension.  By choosing offsets
# that match the appropriate axes limits, the projected contours will sit on
# the 'walls' of the graph
ax.contourf(X, Y, Z, zdir='z', offset=-20, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='x', offset=-40, cmap='coolwarm')
ax.contourf(X, Y, Z, zdir='y', offset=40, cmap='coolwarm')

ax.set(xlim=(-40, 40), ylim=(-40, 40), zlim=(-100, 100),
       xlabel='X', ylabel='Y', zlabel='Z')




#### old and broken



array = [b, c, d]
print("input",np.array(array))
arr_funda = []
quats = []
for ori in array:       
       ori= rod_to_quat(ori)
       quats.append(ori)
       arr_funda.append(ret_to_funda(ori,cub,debug=True))
print("vals",np.array(quats))
arr_funda = np.array(arr_funda)
print("output",arr_funda)

exit(0)
ori = b
ax.quiver(0,0,0,ori[0],ori[1],ori[2])

print("rod ini",ori)
ori= rod_to_quat(ori)
print("quat ini",ori)
ori = ret_to_funda(ori,cub)
print("quat fin",ori)
ori= quat_to_rod(ori)
print("rod fin",ori)

exit(0)

ax.quiver(0,0,0,ori[0],ori[1],ori[2])


