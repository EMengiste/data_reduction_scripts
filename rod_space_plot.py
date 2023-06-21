import os
from ezmethods import *
import pandas as pd
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.pyplot as plt
# Latex interpretation for plots
# Latex interpretation for plots
plt.rcParams.update({'font.size': 5})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 100
plt.rcParams['figure.figsize'] = 20,20

# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1
    

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
ax.set(xlim=(-0.6, 0.6), ylim=(-0.6, 0.6), zlim=(-0.6, 0.6),
       xlabel='X', ylabel='Y', zlabel='Z')

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


simulations = os.listdir(home)
simulations.sort()
simulations.remove("common_files")
home="/media/etmengiste/acmelabpc2_2TB/DATA/jobs/aps/spring_2023/slip_study_rerun/"
for val,col in zip([0,75],["k","r"]):
       sim = fepx_sim("Cube.sim",path=home+simulations[val]+"/Cube.sim")
       step= "28"
       for id in range(500,600):
              stress = sim.get_output("stress",step=step,res="elsets",ids=[id])
              ori = sim.get_output("ori",step=step,res="elsets",ids=[id])
              print(ori)
              stress_mat= to_matrix(stress)
              eig_val, eig_vect = np.linalg.eig(stress_mat)
              print(eig_val)
              pprint(eig_vect)
              ax.scatter(ori[0],ori[1],ori[2])
              start = ori#[0, 0 ,0]
              for ind,eig in enumerate(eig_vect):
                     ax.quiver(start[0],start[1],start[2],
                            start[0]+eig[0],start[1]+eig[1],start[2]+eig[2]
                            ,length=0.05,normalize=True,color=col)

ax.view_init(elev=45., azim=45, roll=0)
plt.savefig("funda_region")
exit(0)

ax.view_init(elev=45., azim=45, roll=0)
plt.show()
exit(0)


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
