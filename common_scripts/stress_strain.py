from fepx_sim import *
from plotting_tools import *

path= os.getcwd()
sim = "simulation.sim"
print(path)

individual_svs(path,sim,outpath=path)