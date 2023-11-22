import tkinter
from tkinter import *
from PIL import Image, ImageTk
import methods
import os
root = Tk()
root.resizable(width=0,height=0)
h=200
chin=300
#
# w width of the canvas
# h height of the canvas
# w:h = 16:9
# aspect = 16/9= 1.777777778
aspect= 4
w = int(aspect*h)
root.geometry(str(w)+"x"+str(h+chin))
print(w,h)
frame1= Frame(root)
frame1.pack()
# Create a photoimage object of the image in the path
image1 = Image.open("img0.png").resize((w,h), Image.ANTIALIAS)
test = ImageTk.PhotoImage(image1)

label1 = tkinter.Label(frame1,image=test)
label1.image = test
# Position image
label1.pack()

#
#
frame= Frame(root)
frame.pack()
#Labels
label = Label(frame1,text=os.getcwd())
label.pack()
label = Label(frame1,text="EZmethod.py")
label.pack()




#Listbox
def tess():        
    for widget in frame.winfo_children():
        widget.destroy()
    label.config(text="Tess")
    #Entries
    name = Entry(frame,width=30)
    #Add some text to begin with
    name.insert(END, string="simulation")
    num_g = Entry(frame,width=30)
    #Add some text to begin with
    num_g.insert(END, string="num_grains")
    run_button = Button(frame,text="Run simulation", 
                        command=lambda:methods.generate_tess(num_g.get(),name.get(),os.getcwd()))
    run_button.pack()
    cancel = Button(frame, text="Cancel",command=action)
    cancel.pack()
    #Gets text in entry
    num_g.pack()
    name.pack()
    
def msh():        
    for widget in frame.winfo_children():
        widget.destroy()
    label.config(text="msh")
    #Entries
    entry = Entry(frame,width=30)
    #Add some text to begin with
    entry.insert(END, string="Some text to begin with.")
    #Gets text in entry
    print(entry.get())
    entry.pack()
    cancel = Button(frame, text="Cancel",command=action)
    cancel.pack()

def run():        
    for widget in frame.winfo_children():
        widget.destroy()
    label.config(text="run")
    cancel = Button(frame, text="Cancel",command=action)
    cancel.pack()

#Buttons
def action():        
    for widget in frame.winfo_children():
        widget.destroy()
    print("Do something")
    label.config(text="Get started!")
    tess_button = Button(frame,text="Generate Tesselation",command=tess)
    tess_button.pack()
    msh_button = Button(frame,text="Generate Mesh", command=msh)
    msh_button.pack()
    run_button = Button(frame,text="Run simulation", command=run)
    run_button.pack()

    button.destroy()

#calls action() when pressed
pressed= False
button = Button(frame,text="Click Me", command=action)
button.pack()

root.mainloop()
exit(0)


#Text
text = Text(height=5, width=30)
#Puts cursor in textbox.
text.focus()
#Adds some text to begin with.
text.insert(END, "Example of multi-line text entry.")
#Get's current value in textbox at line 1, character 0
print(text.get("1.0", END))
text.pack()
#Spinbox
def spinbox_used():
    #gets the current value in spinbox.
    print(spinbox.get())

#Scale
#Called with current scale value.
def scale_used(value):
    print(value)
#Checkbutton
def checkbutton_used():
    #Prints 1 if On button checked, otherwise 0.
    print(checked_state.get())

#Radiobutton
def radio_used():
    print(radio_state.get())


listbox = Listbox(height=4)
fruits = ["Apple", "Pear", "Orange", "Banana"]
for item in fruits:
    listbox.insert(fruits.index(item), item)
listbox.bind("<<ListboxSelect>>", listbox_used)
listbox.pack()
root.mainloop()





spinbox = Spinbox(from_=0, to=10, width=5, command=spinbox_used)
spinbox.pack()
scale = Scale(from_=0, to=100, command=scale_used)
scale.pack()

#Variable to hold on to which radio button value is checked.
radio_state = IntVar()
radiobutton1 = Radiobutton(text="Option1", value=1, variable=radio_state, command=radio_used)
radiobutton2 = Radiobutton(text="Option2", value=2, variable=radio_state, command=radio_used)
radiobutton1.pack()
radiobutton2.pack()

#variable to hold on to checked state, 0 is off, 1 is on.
checked_state = IntVar()
checkbutton = Checkbutton(text="Is On?", variable=checked_state, command=checkbutton_used)
checked_state.get()
checkbutton.pack()
#Import the required libraries
from tkinter import *

#Create an instance of tkinter frame
win= Tk()

#Set the geometry of frame
win.geometry("500x500")

#Get the current screen width and height
screen_width = win.winfo_screenwidth()
screen_height = win.winfo_screenheight()

photo = Tk.PhotoImage(file = "big.gif")
label = Tk.Label(image = photo)
label.pack()
#Print the screen size
print("Screen width:", screen_width)
print("Screen height:", screen_height)

win.mainloop()