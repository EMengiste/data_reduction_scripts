

import numpy as np
def pprint(arr, preamble="\n-+++",max=50):
    space= "                                                   "
    for i in arr:
        i=str(i)
        if isinstance(arr,dict):
            val = preamble+i+": "+str(arr[i])+space
            #                 :boundary_conditions: uniaxial_grip                 '
            print(val[:max]+"|||||------|")
        else:
            val = i+ space
            print(preamble+val[:max]+"|||||------|")
        preamble="    "

def list_sims(list,func="",option=[""]):
    array=[]
    for i in list:
        if func=="":
            print(i.material_parameters[option[0]])
        if func=="stress":
            print(i.get_output("strain",step=option[0],options=option[2])[option[1]])
    return array
##
def plotter(ax,x,y=""):
    if y=="":
        y=np.arange(len(x))
        ax.bar(y,x)
    else:
        ax.plot(x,y)
    return [x,y]
##
def directory_from_list(arr,splitter):
    dict={}
    for i in arr:
        dict[i.split(splitter)[0]]=i.split(splitter)[1]
    return dict
##
#
def plotting_space(x,y=[],layers=3,axis="",resolution=0,ylabel="",debug=False):
    print("-------|---------|---plotting_space--|---------|---------|---------|---------|")
    #print(x,y)
    values = {}
    #pprint(x)
    print("\nlist of vals---\n")
    #print(x)
    num_components= len(x)
    if y==[]:
        #print(num_components)
        wide=1/(len(x)+2)
        for m in range(len(x)):
            i=x[m]
            y= np.arange(len(i))
            spacing=wide*float(m)
            if axis!="":
                axis.bar(y+spacing,i,width=wide,edgecolor="k")
            if debug:

                print("======\n\n")
                print(m)
                print(y+spacing)
                print(float(x.index(i)))
                print("======\n\n")
                for j in range(len(x[0])):                
                    print(str(j)+"  -------|---------|---------|---------|---------|---------|---------|---------|")

                    for i in range(len(x)):
                        ret= str(i)+" |"
                        print(ret)
                        #print(i)
                        spacer="  |"
                        print(x[i][j])
                        while float(x[i][j])>resolution:
                            ret   +="-.--|"
                            spacer+="---+|"
                            x[i][j]=float(x[i][j])-1
                        for _ in range(layers):
                            print(ret)
            axis.set_ylabel(ylabel)
            #print("-------|---------|---------|---------|---------|---------|---------|---------|")

def normailze(array,scale=1,maximum=1,absolute=False,debug=False):
    if debug:
        print("-------|---------|---------|---------|---------|---------|---------|---------|")
        print("-------|---------|---------|---------|---------|---------|---------|---------|")
        print("-------|---------|---------|---------|---------|---------|---------|---------|")

        print(array)
        print("maximum val "+str(maximum)+"------")        
        print("-------|---------|---------|---------|---------|---------|---------|---------|")
        print("-------|---------|---------|---------|---------|---------|---------|---------|")

    # [f(x) if condition else g(x) for x in sequence]
    if isinstance(array[0],str):
        array=[float(i) 
            if "d" not in i 
            else float(i.split("d")[0])
            for i in array ]        
    if isinstance(array[0],list):
        return array
    if absolute:
        array= [abs(i) for i in array]
    if maximum=="":
        maximum = max(array)

    return [scale*i/maximum for i in array]
