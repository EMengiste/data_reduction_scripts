from tool_box import *
from pre_processing import *
import os
import time
##
#
print("1 ", os.getcwd())
os.chdir(path="..")
main_dir=os.getcwd()
print("2 ",os.getcwd())
#
#   print the current working direcotry and list the
#
folders= os.listdir()
pprint(folders)
#
#
#
# add functionality to take direct ff data
experimental_source="path/to/experimental/data"
simulation_tess="/default/simulation"
options ={"morpho":  "gg",
          "mode" : "run",
          "source code" : "neper"}

sim= generate_tess(50,simulation_tess,main_dir,options)
# tess above has destination sim
#
options ={"-order":"2",
          "mode" :"run",
          "source code":"neper"}
generate_msh(sim,4, options)
os.chdir(main_dir)
#
#
options ={"mode" :"run",
          "cores": 4,
          "source code":"fepx"}
destination= "output_data/default/"
print(main_dir+destination+"000")
run_sim(sim[:-4]+"msh", sim[:-4]+"config", destination+"000", options)
#
#
options ={"mode" :"run",
          "source code":"neper"}
post_process(destination+"000",options=options)
# Work on config file and run sim
