import numpy as np
import multiprocessing
import time
from scipy.spatial.transform import Rotation as R
import math

##
def quat_of_angle_ax(angle, raxis):
       # angle axis to quaternion
       #
       half_angle = 0.5*angle
       #
       cos_phi_by2 = math.cos(half_angle)
       sin_phi_by2 = math.sin(half_angle)
       #
       rescale = sin_phi_by2 / np.sqrt(np.dot(raxis,raxis))
       quat = np.append([cos_phi_by2],np.tile(rescale,[3])*raxis)
       if cos_phi_by2<0:
              quat = -1*quat
       #
       #
       return quat
##
def Cubic_sym_quats():
    # Generate Cubic symetry angle axis pairs for the cubic fundamental region
    pi = math.pi
    AngleAxis =  np.array([[0.0     , 1 ,   1,    1 ],   # % identity
                    [pi*0.5  , 1 ,   0,    0 ],   # % fourfold about x1
                    [pi      , 1 ,   0,    0 ],   #
                    [pi*1.5  , 1 ,   0,    0 ],   #
                    [pi*0.5  , 0 ,   1,    0 ],   # % fourfold about x2
                    [pi      , 0 ,   1,    0 ],   #
                    [pi*1.5  , 0 ,   1,    0 ],   #
                    [pi*0.5  , 0 ,   0,    1 ],   # % fourfold about x3
                    [pi      , 0 ,   0,    1 ],   #
                    [pi*1.5  , 0 ,   0,    1 ],   #
                    [pi*2/3  , 1 ,   1,    1 ],   # % threefold about 111
                    [pi*4/3  , 1 ,   1,    1 ],   #
                    [pi*2/3  ,-1 ,   1,    1 ],   # % threefold about 111
                    [pi*4/3  ,-1 ,   1,    1 ],   #
                    [pi*2/3  , 1 ,  -1,    1 ],   # % threefold about 111
                    [pi*4/3  , 1 ,  -1,    1 ],   #
                    [pi*2/3  ,-1 ,  -1,    1 ],   # % threefold about 111
                    [pi*4/3  ,-1 ,  -1,    1 ],   #
                    [pi      , 1 ,   1,    0 ],   # % twofold about 110
                    [pi      ,-1 ,   1,    0 ],   #
                    [pi      , 1 ,   0,    1 ],   #
                    [pi      , 1 ,   0,   -1 ],   #
                    [pi      , 0 ,   1,    1 ],   #
                    [pi      , 0 ,   1,   -1 ]])

    cubic_sym = np.array([quat_of_angle_ax(a[0],a[1:]) for a in AngleAxis])
    return cubic_sym
#
#
###
def quat_prod(q1, q2):
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
       quat = np.array([ind1,v[0],v[1],v[2]])
       #
       if quat[0]<0:
              quat=-1*quat
       #print(quat)
       return quat
       #

def quat_prod_multi(q1,q2):
	quat =[]
	for i in range(len(q1)):
		quat.append(quat_prod(q1[i],q2[i]))
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
#
##
def dif_degs(start,fin,debug=False):
    # Get basis mats
    q_ij,q_ji = rot_mat(start,fin)
    # Get misorination mat
    v1 = normalize_vector(R.from_matrix(q_ij).as_quat())
    #v1 = normalize_vector(rot_to_quat_function(q_ij,option=""))
    # Get fuda
    #r1 = ret_to_funda(v1)
    r1 = v1
    # Get degs
    thet_ij =math.degrees(math.acos(min([r1[0],1])))
    if debug:
        print(np.dot(q_ij.T,start[0]))
        print("---")
        print(np.dot(q_ij.T,fin[0]))
        r2 = ret_to_funda(R.from_matrix(q_ji).as_quat())
        thet_ji =math.degrees(math.acos(r2[0]))
        return [thet_ij,thet_ji]
    
    return thet_ij
###
def rot_mat(arr1,arr2):
    #   Find the roation matrix from a basis matrix 
    #       Q_ij = arr1 => arr2
    #       Q_ji = arr1 => arr2
    R_ij = []
    R_ji = []
    if len(arr1) ==len(arr2):
        for a in arr1:
            temp = []
            for b in arr2:
                    temp.append(np.dot(a,b))
            R_ij.append(temp)
        for b in arr1:
            temp = []
            for a in arr2:
                    temp.append(np.dot(a,b))
            R_ji.append(temp)                     
        return [np.array(R_ij),np.array(R_ji)] 
    else:  
            print("not same size")

def quaternion_misorientation(q1,q2):
       # https://gitlab.tudelft.nl/-/snippets/190
       # Input:
       #       q1 = [w1,x1,y1,z1]
       #       q2 = [w2,x2,y2,z2]
       # Output: 
       #        Theta
       print(len(q1))
       w1,x1,y1,z1 = q1
       w2,x2,y2,z2 = q2
       theta = 2*math.acos(w1*w2 +x1*x2 + y1*y2 + z1*z2)
       return theta
##
def ret_to_funda(quat="",rod="", sym_operators=Cubic_sym_quats(),debug=False):
       #    Return quaternion to the fundamental region given symerty 
       #        operatiors
       #
       if str(rod)!="":
              quat = rod_to_quat(rod)
       #
       m = len(sym_operators)
       n = 1
       # if passing a set of symetry operators make sure to [quat]
       tiled_quat = np.tile(quat,( m,1))
       #
       reshaped_quat=tiled_quat.reshape(m*n,4,order='F').copy()
       #sym_operators=sym_operators.T
       equiv_quats = quat_prod_multi(reshaped_quat,np.tile(sym_operators,(1,n)))
       return equiv_quats
#
#pool = multiprocessing.Pool(processes=90)
#
q1 = [.1, 0.2,0.3,0.4]
q1INV = [.1, -0.2,-0.3,-0.4]
q1= ret_to_funda(q1)
q1INV= ret_to_funda(q1INV)
q2 = [0.5, 0.6,0.7,0.8]
#q2= ret_to_funda(q2)
q2= normalize_vector(q2)

print(quat_prod(q1,q1INV))

print(quaternion_misorientation(q2,q2))
print(quaternion_misorientation(q1,q1INV))

exit(0)
tic = time.perf_counter()
#value = pool.map(quat_prod,*(q1,q2))
value=quat_prod(q1,q1)
print(value)
toc = time.perf_counter()
print(f"Ran the code in {toc - tic:0.4f} seconds")
