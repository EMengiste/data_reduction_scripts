# Import Module
from tkinter import *
from PIL import Image, ImageTk
import pre_processing as methods
import os
height = 400
aspect = 2.703296703
width = int(aspect* height)

home_path= os.path.dirname(os.path.realpath(__file__))


# Create Tkinter Object
root = Tk()
top = Frame(root)
bottom= Frame(root)
top.pack()
bottom.pack()
 

# Read the Image
image = Image.open(home_path+"/thre_plot_test.png")


# Resize the image using resize() method
resize_image = image.resize((width, height))
 
img = ImageTk.PhotoImage(resize_image)
 
# create label and add resize image
label1 = Label(top, image=img)
label = Label(top,text=os.getcwd())
label.pack()
label1.image = img
label1.pack()


#   Tesselation generation
def tess_screen():        
    for widget in bottom.winfo_children():
        widget.destroy()
    #Entries
    name = Entry(bottom, text= "Name",width=30)
    #Add some text to begin with
    name.insert(END, string="simulation")
    num_g = Entry(bottom, text= "Number of",width=30)
    #Add some text to begin with
    num_g.insert(END, string="num_grains")
    run_button = Button(bottom,text="Run simulation", 
                        command=lambda:methods.generate_tess(num_g.get(),name.get(),os.getcwd()))
    run_button.pack()
    cancel = Button(bottom,text="cancel",command=clear_bottom)
    cancel.pack()
    #Gets text in entry
    num_g.pack()
    name.pack()

#   Mesh generation
def mesh_screen():        
    for widget in bottom.winfo_children():
        widget.destroy()
    #Entries
    name = Entry(bottom, text= "Name",width=30)
    #Add some text to begin with
    name.insert(END, string="simulation")
    num_g = Entry(bottom, text= "Number of",width=30)
    #Add some text to begin with
    num_g.insert(END, string="num_grains")
    run_button = Button(bottom,text="Run simulation", 
                        command=lambda:methods.generate_tess(num_g.get(),name.get(),os.getcwd()))
    run_button.pack()
    cancel = Button(bottom,text="cancel",command=clear_bottom)
    cancel.pack()
    #Gets text in entry
    num_g.pack()
    name.pack()
    

def clear_bottom():    
    for widget in bottom.winfo_children():
        widget.destroy()
    tess = Button(bottom,text="Generate Tess",command=tess_screen)
    msh = Button(bottom,text="Generate Mesh")
    config = Button(bottom,text="Generate Config")
    run = Button(bottom,text="Run Simulation")
    analize = Button(bottom,text="Analize Data")
    test_matrix = Button(bottom,text="Generate Test Matrix")
    cancel = Button(bottom,text="quit",command=quit)
    tess.pack()
    msh.pack()
    config.pack()
    run.pack()
    analize.pack()
    test_matrix.pack()
    cancel.pack()



tess = Button(bottom,text="Generate Tess",command=tess_screen)
msh = Button(bottom,text="Generate Mesh",command=mesh_screen)
config = Button(bottom,text="Generate Config")
run = Button(bottom,text="Run Simulation")
analize = Button(bottom,text="Analize Data")
test_matrix = Button(bottom,text="Generate Test Matrix")
cancel = Button(bottom,text="cancel",command=clear_bottom)
tess.pack()
msh.pack()
config.pack()
run.pack()
analize.pack()
test_matrix.pack()
cancel.pack()
#
#




#
# Execute Tkinter
root.mainloop()