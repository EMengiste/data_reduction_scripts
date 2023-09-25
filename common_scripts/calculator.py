import numpy as np
##
def quat_prod(q1, q2,debug=False):
       # Quaternion Product
       # input a q1 and 4*1
       many =False
       try:
              a = np.array([q[0] for q in q1.T])
              b = np.array([q[0] for q in q2.T])
              avect= np.array([q[1:] for q in q1.T])
              bvect= np.array([q[1:] for q in q2.T])
              if debug:
                     print("a",avect.shape)
                     print("b",bvect.shape)
              many=True
       except:
              a=q1[0]
              b=q2[0]
              avect=q1[1:]
              bvect=q2[1:]
       #
       a3 = np.tile(a,[3,1]).T
       b3 = np.tile(b,[3,1]).T

       if debug:
              print("a3",a3.shape)
       #
       if many:
              dotted_val = np.array([np.dot(a,b) for a,b in zip(avect,bvect)])
              crossed_val = np.array([np.cross(a,b) for a,b in zip(avect,bvect)])
       else:
              dotted_val=np.dot(avect,bvect)
              crossed_val=np.cross(avect, bvect)
       
       #
       ind1 =a*b - dotted_val
       if debug:
              print(ind1.shape)

       #print(crossed_val.shape)
       #print((b3*avect).shape)
       val=a3*bvect + b3*avect +crossed_val

       if debug:
              print(val)
       quat = np.array([[i,v[0],v[1],v[2]] for i,v in zip(ind1, val)])
       #
       if many:
              max = 0
              max_ind=0
              for ind,q in enumerate(quat):
                     if q[0]<0:
                            quat[ind] = -1*quat[ind]
                     if ind==0:
                            max= quat[ind][0]
                            max_ind=ind
                     elif quat[ind][0]>max:
                            #print("larger")
                            #print(quat[ind])
                            max=quat[ind][0]
                            max_ind=ind
                     #
              value = normalize_vector(quat[max_ind])
              #print("--------",value)
              return value
       else:
              if quat[0]<0:
                     quat=-1*quat
              #print(quat)
              return quat
#
q1 = [.1, 0.2,0.3,0.4]
q2 = [0.5, 0.6,0.7,0.8]
print(quat_prod(q1,q2,debug=True))