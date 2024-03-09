import pandas as pd 
import h5py
import sys
import os
import numpy as np
import math
import time


preamble = "======--"
preamble2 = "\n--"

def to_deg(arr):
    temp = []
    for i in arr:
        temp.append(math.degrees(i))
    return temp

sim_path = sys.argv[1]

files_in_dir = [i for i in os.listdir() if i.endswith(".h5oina")]
print(files_in_dir)
write_file = False
tic = time.perf_counter()
print("starting code")

for file in files_in_dir:

    in_file =  h5py.File(file,"r")
    file = file.replace(" ","_")[:-7]
    print(f" Processing file {file}")
    print(in_file["/1/EBSD/Data"].keys())
    # print()
    X = in_file["/1/EBSD/Data/X/"]
    Y = in_file["/1/EBSD/Data/Y/"]
    euler = in_file["/1/EBSD/Data/Euler"]
    num_vals= len(euler[:])
    #
    max_val = max(X[:])
    size_x = np.where(X[:] ==max_val)[0][0]+1
    size_y = int(num_vals/size_x)
    #
    vox_size_x = X[1]-X[0]
    vox_size_y = Y[2*size_x]-Y[size_x]
    #
    print(f"  {size_x} {size_y}\n")
    print(f"  {vox_size_x} {vox_size_y}\n")
    exit(0)
    if write_file:
        tesr_file= open(file+".tesr","w")
        tesr_file.write("***tesr\n")
        tesr_file.write(" **format\n")
        tesr_file.write("   2.1\n")
        tesr_file.write("**general\n")
        tesr_file.write("  2\n")
        tesr_file.write(f"  {size_x} {size_y}\n")
        tesr_file.write(f"  {vox_size_x} {vox_size_y}\n")
        tesr_file.write(" **oridata\n")
        tesr_file.write("   euler-bunge\n")
        tesr_file.write("   ascii\n")
        for i in range(num_vals):
            tesr_file.write(f"   {str(to_deg(euler[i]))[1:-1]}\n")
        tesr_file.write("***end\n")
        tesr_file.close()
    os.system(f"neper -V {file}.tesr -datavoxcol ori -print {file}_ebsd")
    # try:
    #     quality= in_file["/1/EBSD/Data/Pattern Quality/"][::1]
    #     phase= in_file["/1/EBSD/Data/Phase/"][::1]
    #     error= in_file["/1/EBSD/Data/Error/"][::1]
    # except:
    #     continue
    # ax.scatter(X,Y,c=quality,s=.1)
    # ax2.scatter(X,Y,c=phase,s=.1)
    # ax3.scatter(X,Y,c=error,s=.1)
    # fig.savefig(file,dpi=100)
    # fig.clf()

toc = time.perf_counter()
print("===")
print("===")
print(f"Generated data in {toc - tic:0.4f} seconds")
exit(0)
trim=None
writing=False
for filename in files_in_dir:
    in_file =  ""#h5py.File(filename,"r")
    print(in_file["/1/"].keys())
    out_name = filename[:-7].replace(" ","_")

    # os.system("convert +append "+out_name+"_100.png "+out_name+"_110.png "+out_name+"_111.png "+out_name+".png")
    exit(0)
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