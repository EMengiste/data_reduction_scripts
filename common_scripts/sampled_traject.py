
import os
#
path = os.getcwd()
dir_name= path[path.rindex('/')+1:]
#
#
sample_num = 100
start = 500
#
#reports = open(path+'/post.report','r')
num_steps = 27#int(reports.readlines()[8][15:])

print(num_steps,"======")
#
path +="/results/elsets/ori/ori.step"
init= open('ini','w')
step_0 = open(path+'0','r').readlines()

for i in range(start,start+sample_num):
    init.write(step_0[i])


final= open('fin', 'w')
step_n = open(path+str(num_steps),'r').readlines()
for i in range(start,start+sample_num):
    final.write(step_n[i])

all= open("all","w")
all.close()
all= open('all','a+')
for j in range(num_steps+1):
    print(path+str(j))
    curr= open(path+str(j),'r').readlines()
    for i in range(start,start+sample_num):
        all.write(curr[i])