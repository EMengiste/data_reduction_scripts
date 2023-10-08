import os
from itertools import chain, combinations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import matplotlib.animation as animation
import numpy as np
from scipy.linalg import polar
from scipy.spatial.transform import Rotation as Rot
import math

from matplotlib.backends.backend_agg import FigureCanvasAgg
from io import BytesIO

#https://stackoverflow.com/a/36376310

# This outputs PNG images but other formats are available (print_jpg or print_tif, for instance).
SIZE=10
# Latex interpretation for plots
plt.rcParams.update({'font.size': SIZE})
#plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"
plt.rcParams["figure.subplot.left"] = 0.045
plt.rcParams["figure.subplot.bottom"] = 0.08
plt.rcParams["figure.subplot.right"] = 0.995
plt.rcParams["figure.subplot.top"] = 0.891
plt.rcParams["figure.subplot.wspace"] = 0.21
plt.rcParams["figure.subplot.hspace"] = 0.44
plt.rcParams['figure.figsize'] = 8,8 #
plt.rc('font', size=SIZE)            # controls default text sizes
plt.rc('axes', titlesize=SIZE)       # fontsize of the axes title
plt.rc('axes', labelsize=SIZE)       # fontsize of the x and y labels
plt.rc('xtick', labelsize=SIZE)      # fontsize of the tick labels
plt.rc('ytick', labelsize=SIZE)      # fontsize of the tick labels
plt.rc('legend', fontsize=SIZE)      # legend fontsize
plt.rc('figure', titlesize=SIZE)     #
#
#
#
# Initial node coordinates in order
#
def nodes_15(): 
   fig,axs = plt.subplots(1,1,subplot_kw=dict(projection='3d'))
   X= [0  ,1,1.1]
   Y= [0.1,0,0.5]  
   axs.plot(X,Y)
   plt.show()

nodes_15()