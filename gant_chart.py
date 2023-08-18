import pandas as pd
import matplotlib.pyplot as plt 

file = open("post.force.z1").readlines()[2:]
strain_rate = 0.001
stress = []
strain = []
for i in file:
    arr = i.split()[-3:]
    print("string version",arr)
    
    arr = [float(n)  for n in arr]
    print("float",arr)
    stress.append(arr[0]/arr[1])
    strain.append(arr[2]*strain_rate)

plt.plot(strain,stress)
plt.show()