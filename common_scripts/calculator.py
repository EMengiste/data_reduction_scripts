import numpy as np
import multiprocessing
import time
##
def normalize_vector(vect,magnitude=False):
    value= 0
    final = vect
    for i in vect:
        value+=(i**2)
    mag=(value)**0.5
    for i in range(len(vect)):
        final[i] = final[i]/mag
    if magnitude:
        return [final,mag]
    else:
        return final
  #
 #
##
def quat_prod(q1, q2,debug=False):
       # Quaternion Product
       # input a q1 and 4*1
       print(q1,q2)
       a=q1[0]
       b=q2[0]
       avect=q1[1:]
       bvect=q2[1:]
       #
       a3 = np.tile(a,[3,]).T
       b3 = np.tile(b,[3,]).T
       #
       dotted_val=np.dot(avect,bvect)
       crossed_val=np.cross(avect, bvect)
       #
       ind1 =a*b - dotted_val
       print(ind1.shape)
       val=a3*bvect + b3*avect +crossed_val
       quat = [ind1,val[0],val[1],val[2]]
       #
       if quat[0]<0:
              quat=-1*quat
       quat = normalize_vector(quat)
       return quat
#
q1 = np.array([0.1, 0.2,0.3,0.4])
q2 = np.array([0.5, 0.6,0.7,0.8])
print(quat_prod(q2,q1,debug=True))

#num_calcs = 10
#q1_array = [ q1 for _ in range(num_calcs)]
#q2_array = [ q2 for _ in range(num_calcs)]


#pool = multiprocessing.Pool(processes=90)
#value = pool.map(quat_prod,(q1_array,q2_array))