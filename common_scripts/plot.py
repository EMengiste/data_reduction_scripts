import os
from tkinter import *
import matplotlib as mpl
from fepx_sim import *
from tool_box import *

mpl.rcParams.update({'font.size': 35})
mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'DejaVu Serif'
mpl.rcParams["mathtext.fontset"] = "cm"
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg,
NavigationToolbar2Tk)
import pandas as pd
import numpy as np
#from sklearn.linear_model import LinearRegression
# plot function is created for
# plotting the graph in
# tkinter window
#

debug=0
#
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
#
#
class SVS_plot: ##
    def __init__(self, names, path):
        self.names= names
        self.path = path

    def display(self,ax,show_yield=True,show_offset=True):
        num= len(self.names)
        name= self.names
        for i in range(num):
            print(name[i])
            #
            simulation = fepx_sim(name[i],path=self.path+"/"+name[i])
            # Check if simulation value is available 
            # if not post process and get
            try:
                stress=simulation.get_output("stress_eq",step="all")
                strain=simulation.get_output("strain_eq",step="all")
            except:
                simulation.post_process(options="-resmesh stress_eq,strain_eq")
                stress=simulation.get_output("stress_eq",step="all")
                strain=simulation.get_output("strain_eq",step="all")
            plot_stress_strain(ax,stress,strain)

            # calculate the yield values
            yield_values = find_yield(stress,strain)
            ystrain,ystress =yield_values["y_strain"],yield_values["y_stress"]
            stress_off = yield_values["stress_offset"]
            if show_yield: ax.plot(ystrain,ystress,"k*",ms=20,label="$\sigma_y$="+str(yield_values["y_stress"]))
            # if show_offset: ax.plot(strain,stress_off,"ko--",ms=5)
            ax.legend()
    
            #

#
#
stress_sim= '$\sigma_{eq}$'
strain_sim= '$\epsilon_{eq}$'
stress_exp = '$\sigma$'
strain_exp="$\\v"+"arepsilon$"
x_label = f'{strain_exp} (\%)'
y_label = f'{stress_exp} (MPa)'
#
#
#
#
def plot():
    plt.clf()
    # the figure that will contain the plot
    fig = plt.Figure(figsize = (11, 10),
                 dpi = 100)
    # adding the subplot
    plot1 = fig.add_subplot(111)
    if e2.get()=='':
        upper_bound= 400
    else:
        upper_bound= float(e2.get())
    # plotting the graph
    list= SVS_plot([listbox.get(i) for i in listbox.curselection()],"./")
    #,150,200,250,ms=ms)
    list.display(plot1)

    # creating the Tkinter canvas
    #fig.supxlabel(x_label)
    #fig.supylabel(y_label)
    fig.tight_layout()
    fig.subplots_adjust(left=0.15, right=0.995,
                          top=0.95, bottom=0.12, wspace=0.035)
    entry= e1.get()

    if entry!= '':
        fig.savefig(entry+".png", dpi=800, bbox_inches="tight")

    # containing the Matplotlib figure
    canvas = FigureCanvasTkAgg(fig,
                               master = window)

    canvas.draw()

    # placing the canvas on the Tkinter window
    canvas.get_tk_widget().grid(row = 0, column = 1, sticky = N, rowspan=3, pady = 2)

    # creating the Matplotlib toolbar
    # placing the toolbar on the Tkinter window
    canvas.get_tk_widget().grid(row = 0, column = 1, sticky = N, rowspan=3, pady = 2)
def end_plotting():
    window.destroy
    exit(0)

# the main Tkinter window
window = Tk()

# setting the title
window.title('Plotting in Tkinter')

# dimensions of the main window
window.geometry("1450x1000")

# button that displays the plot
plot_button = Button(master = window,
                     command = plot,
                     height = 2,
                     width = 10,
                     text = "Plot")
l1 = Label(text="File Name")
e1 = Entry(window)
l2 = Label(text="Y Upper bound")
e2 = Entry(window)
bt = Button(window, text="Quit", command=end_plotting)
CSVs = []
curr_path = os.getcwd()
for i in os.listdir(curr_path):
    CSVs.append(i)
print(CSVs.sort())
csv = StringVar(value=CSVs)
#
print(csv)
listbox = Listbox(
    window,
    listvariable=csv,
    height=30,
    selectmode=MULTIPLE)
#
# place the button
# in main window
listbox.grid(row = 0, column = 0, sticky = W, pady = 2)
plot_button.grid(row = 0, column = 2, sticky = E, pady = 2)
e1.grid(row=1,column=3,sticky= E)
e2.grid(row=2,column=3,sticky= E)
l1.grid(row=1,column=2,sticky= E)
l2.grid(row=2,column=2,sticky= E)
bt.grid(row=2,column=3,sticky= E)
for widget in window.winfo_children():
    print(f"{widget.widgetName}\n\n")
# run the gui
window.protocol("WM_DELETE_WINDOW", end_plotting)
window.mainloop()
