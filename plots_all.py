import os
from ezmethods import *
import pandas as pd
import matplotlib.pyplot as plt
import os
# Latex interpretation for plots
# Latex interpretation for plots
plt.rcParams.update({'font.size': 35})
plt.rcParams['text.usetex'] = True
plt.rcParams['font.family'] = 'DejaVu Serif'
plt.rcParams["mathtext.fontset"] = "cm"#
plt.rcParams["figure.dpi"] = 400
plt.rcParams['figure.figsize'] = 27,9

# left=0.08, right=0.98,top=0.9, bottom=0.2, wspace=0.02, hspace=0.1
    

plot_eff_strain(0,all=True,marker_size=15)
#plot_yield_stress(0,all=True,marker_size=15)

exit(0)