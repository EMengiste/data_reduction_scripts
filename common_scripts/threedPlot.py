import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

SIZE=12

#
plt.rc('text',usetex=True)
plt.rc('font', family='DejaVu Sans', size=SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)    # legend fontsize
plt.rc('figure', titlesize=SIZE)  #
plt.rcParams['figure.figsize'] = 3,3

fig = plt.figure()
# 
ax= fig.add_subplot(projection="3d")
# set viewing angle
ax.view_init(elev=30,azim= 15)

## plot image
ax.plot([0,1],[0,1],[0,1],"k")



#
ax.xaxis.pane.fill=False
ax.yaxis.pane.fill=False
ax.zaxis.pane.fill=False

#https://stackoverflow.com/a/50435041
ax.xaxis.pane.set_edgecolor('black')
ax.yaxis.pane.set_edgecolor('black')
ax.zaxis.pane.set_edgecolor('black')
#
ax.set_xlim([0,1])
ax.set_ylim([0,1])
ax.set_zlim([0,1])
#
ax.grid(False)
ax.set_aspect("equal")
ax.set_xlabel("X",labelpad=5)
ax.set_ylabel("Y",labelpad=5)
ax.set_zlabel("Z",labelpad=5)

fig.subplots_adjust(left=0.15,bottom=0.2)
# plt.tight_layout()
fig.savefig("thre_plot_test",dpi=600)