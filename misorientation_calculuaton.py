import math
import numpy as np
pi = math.pi
#
def quat_of_angle_ax(angle, raxis):
    #    Angle axis to quaternion conversion
    #        
    # ------ Input
    #
    # angle : rotation angle
    # raxis : axis of rotation  
    #
    # ------ Output
    #
    #  quat : quaternion representation of input 
    #
    half_angle = 0.5*angle
    #
    cos_phi_by2 = math.cos(half_angle)
    sin_phi_by2 = math.sin(half_angle)
    #
    rescale = sin_phi_by2 / np.sqrt(np.dot(raxis,raxis))
    quat = np.append([cos_phi_by2],np.tile(rescale,[3])*raxis)
    #
    if cos_phi_by2<0:
        quat = -1*quat
    #
    return quat
    #
  #
 #
#

def quat_prod(q1, q2):
    #    Quaternion product calculation
    #        
    # ------ Input
    #
    # q1 : first quaternion
    # q2 : second quaternion  
    #
    # ------ Output
    #
    # quat: product of q1 and q2
    #
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
    return quat
    #
  #
 #
#

def quat_misori(q1, q2, degrees=True):   
    #    Quaternion misorientation calculation
    #        (in degrees by default)
    #
    # ------ Input
    #
    # q1 : first quaternion
    # q2 : second quaternion  
    #
    # ------ Output
    #
    # angle : misoriention angle
    #
    q1[0]=-q1[0]
    product = quat_prod(q1,q2)
    #
    ind1=product[0]
    #
    angle = 2*math.acos(min(1,ind1))
    #
    if degrees:
        return math.degrees(angle)
    else:
        return angle

def quat_prod_funda(q1,q2):     
    #    Quaternion product calculation for returning 
    #    to the fundamental region of rodrigues space
    #        
    # ------ Input
    #
    # q1 : input orientation in quaternions
    # q2 : symetry operators in quaternions   
    #
    # ------ Output
    #
    # quat_max: Symetrically unique version of q1
    #
	quat =[]
    #
	for i in range(len(q1)):
		quat.append(quat_prod(q1.T[i],q2.T[i]))
    #
	max = 0
	max_ind=0
    #
	for ind,q in enumerate(quat):
		if q[0]<0:
			quat[ind] = -1*quat[ind]
        #
		if ind==0:
			max= quat[ind][0]
			max_ind=ind
            #
		elif quat[ind][0]>max:
			max=quat[ind][0]
			max_ind=ind
		#
	quat_max = normalize_vector(quat[max_ind])
    #
	return quat_max

def hex_sym():
    #   Generator for hexagonal symmetry operators in angle axis
    #
    # ------ Output
    #   AngleAxis: angle axis representation of symmetry operators
    #               for hexagonal symmetry
    #
    p3 = math.pi/3
    p6 = math.pi/6
    sixfold = []
    twofold = []
    for i in range(6):
        arr1 = [pi,math.cos(p6*i),math.sin(p6*i),0]
        twofold.append(arr1)
        arr2 = [p3*i, 0,0,1]
        sixfold.append(arr2)
    #
    AngleAxis = sixfold+twofold
    #
    return np.array(AngleAxis)

def ret_to_funda(quat, sym_operators=hex_sym(),debug=False):
    #    Return quaternion to the fundamental region given symmetry
    #    operators
    #
    # ------ Input
    #   quat: quaternion orientation
    #   sym_operators(optional): symetry operators for a given crystal symmetry
    #                            default set for hexagonal symmetry 
    # ------ Output
    #   equiv_quats: quaternion within the fundamental region of rodrigues space
    #                for a given crystal symmetry
    #
    m = len(sym_operators)
    n = 1
    # if passing a set of symetry operators make sure to [quat]
    tiled_quat = np.tile(quat,(1,m))
    #
    reshaped_quat=tiled_quat.reshape(4,m*n,order='F').copy()
    #
    sym_operators=sym_operators.T
    equiv_quats = quat_prod_funda(reshaped_quat,np.tile(sym_operators,(1,n)))

    return equiv_quats
#
def normalize_vector(vect):
    # Normalize vector
    value= 0
    final = vect
    for i in vect:
        value+=(i**2)
    mag=(value)**0.5
    for i in range(len(vect)):
        final[i] = final[i]/mag
    return final
  #
 #
#


