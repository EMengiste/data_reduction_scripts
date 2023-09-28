import numpy as np
import multiprocessing
import time
##
def quat_prod(q1, q2,debug=False):
       # Quaternion Product
       # input a q1 and 4*1
       a=q1[0]
       b=q2[0]
       avect=np.array(q1[1:])
       bvect=np.array(q2[1:])
       #
       dotted_val=np.dot(avect,bvect)
       crossed_val=np.cross(avect, bvect)
       #
       ind1 =  a* b - dotted_val
       v    =  a*bvect + b*avect +crossed_val
       #
       if debug:
              print(v)
       quat = np.array([ind1,v[0],v[1],v[2]])
       #
       if quat[0]<0:
              quat=-1*quat
       #print(quat)
       return quat
       #


pool = multiprocessing.Pool(processes=90)
#
q1 = [.1, 0.2,0.3,0.4]
q2 = [0.5, 0.6,0.7,0.8]

tic = time.perf_counter()
value = pool.map(quat_prod,*(q1,q2))
toc = time.perf_counter()
print(f"Ran the code in {toc - tic:0.4f} seconds")
