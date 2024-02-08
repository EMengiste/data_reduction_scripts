import pandas as pd 
import h5py
import os
import math
def to_deg(arr):
    temp = []
    for i in arr:
        temp.append(math.degrees(i))
    return temp
files_in_dir = [i for i in os.listdir() if i.endswith(".h5oina")]
print(files_in_dir)
# in_file =  h5py.File(files_in_dir[0],"r")
# print(in_file["/1/EBSD/Data"].keys())
# print(in_file["/1/EBSD/Header/Project Notes"][:])
trim=None
writing=False
for filename in files_in_dir:
    in_file =  ""#h5py.File(filename,"r")
    out_name = filename[:-7].replace(" ","_")
    os.system("convert +append "+out_name+"_100.png "+out_name+"_110.png "+out_name+"_111.png "+out_name+".png")
    # exit(0)
    if writing:
        print(out_name)
        out_file =open(out_name,"w")
        for i in in_file["/1/EBSD/Data/Euler"][::trim]:
            i=to_deg(i)
            out_file.writelines(str(i).replace(",", " ")[1:-1]+"\n")
        out_file.close()
        in_file.close()
        file="neper -V 'pts(type=ori):file("+out_name+",des=euler-bunge)' "
        os.system(file+ "-space pf -pfmode density -dataptsscale 0:25 -pfpole 1:0:0 -print "+out_name+"_100 -pfpole 1:1:0 -print "+out_name+"_110 -pfpole 1:1:1 -print "+out_name+"_111")

exit(0)


def read_ebsd_data_ctf(file_name):
    file_content= open(file_name).readlines()[:14]
    xdim= ydim = 0
    for i in file_content:
        if i.startswith("XCells"):
            xdim = int(i.split()[1])
        if i.startswith("YCells"):
            ydim = int(i.split()[1])
    print(xdim,ydim)
    df=pd.read_csv(file_name,skiprows=(range(14)),sep="\t")
    return df

df = read_ebsd_data_ctf("20 hr.ctf")
oris =df[["Euler1","Euler2","Euler3"]]
coos =df[["X","Y"]]
output_file = open()