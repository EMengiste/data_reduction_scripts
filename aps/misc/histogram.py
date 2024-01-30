import numpy as np
from scipy.stats import lognorm
import matplotlib.pyplot as plt
import statistics as stats
plt.rcParams.update({'font.size': 15})
plt.rcParams['text.usetex'] = True

#
#
# read lines of the files and put them into arrays
file1= open("radii", "r")
file2= open("stat.stcell","r")
arr1= []
arr2= []

for i in file1.readlines():
    arr1.append(float(i))
mean=(stats.mean(arr1))

for i in file2.readlines():
    arr2.append(float(i))
mean1=(stats.mean(arr2))

print(np.mean(np.log(arr1)),"mean")
print(np.std(np.log(arr1)), "sigma")
#
#
#
# Save values of the sigma and mean
with open("values", "w") as f:
    s, loc, scale = lognorm.fit(arr1)
    f.write(str(mean)+" mean ")
    f.write(str(s)+" sigma ")
    f.write(str(loc)+" loc ")
    f.write(str(scale)+" scale\n\n")
    s, loc, scale = lognorm.fit(arr2)
    f.write(str(mean1)+" mean ")
    f.write(str(s)+" sigma ")
    f.write(str(loc)+" loc ")
    f.write(str(scale)+" scale")
#
#
#
# Normalizes data so mean is at 1
for i in range(len(arr1)):
    arr1[i]= arr1[i]/(mean)

for i in range(len(arr2)):
    arr2[i]= arr2[i]/(mean1)

data = arr1
data1= arr2
#
#
#
# Create subplots for the overlaping graphs
fig, ax = plt.subplots(1, 1)
ax2 = ax.twinx() # makes second y label
#
#
#
# Plot the histogram.
ax.hist(data, bins= 15 ,density=True, alpha=0.3,edgecolor= 'black', linewidth=1.2, color='k',label='Experimental distribution')
#ax.hist(data1, bins= 15 ,density=True, alpha=0.5, edgecolor= 'black',linewidth=1.4, color='k',label='Representative distribution')
ax.legend()
#
xmin, xmax = plt.xlim()
#
#
# Write lognormal line for Experimental Data
s, loc, scale = lognorm.fit(data)
x = np.linspace(xmin, xmax, 1000)
p = lognorm.pdf(x, s)
#
#
# Write lognormal line for Simulated data
s1, loc1, scale1 = lognorm.fit(data1)
x1 = np.linspace(xmin, xmax, 1000)
p1 = lognorm.pdf(x1, s1)
#
#
#
# Plot lognormal fit lines
ax2.plot(x, p, 'k--', lw=1,label='Experimental Lognormal fit')
ax2.plot(x1, p1, 'k.--', lw=1,label='Simulated Lognormal fit')

#
#
#
# Set labels for each graph and axies and save the figure
ax.set_xlabel(r'\textbf{Normalized diameters}')
ax.set_ylabel(r'\textbf{Count}',)
ax2.set_ylabel(r'\textbf{Probability Density Function}')
plt.xlim([min(data),max(data)])
ax.legend(loc="upper right")
plt.title("mean = "+ str(s1))
plt.tight_layout()
plt.savefig("staty3.png")
#plt.show()
