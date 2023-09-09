import math

def tau_cut(vol_frac, rad, b_p):
    # a : shearing contribution
    # vol_frac: volume fraction of precipitate phase
    # rad: average radius of precipitate phase
    # b_p: burgers vector of precipitate phase
    tau_precip = ((vol_frac*rad)/b_p)**(0.5)
    return tau_precip
#
#
def tau_bow(c, shear_mod, vol_frac, rad, b_m):
    # c: shear modulus baias term
    # shear_mod: shear modulus of matrix
    # vol_frac: volume fraction of precipitate phase
    # rad: average radius of precipitate phase
    # b_m: burgers vector of martix phase
    L_2r = rad * ((math.pi**0.5) - 2*(vol_frac**0.5))
    tau_precip = (c * shear_mod * b_m)/L_2r
    return tau_precip
#
#
def write_precip_file(frac,rad,name="simulation",res="Elset"):
    ####    Open precip distribution file and write header
    precip_dist_file = open(name+".precip",'w')
    precip_dist_file.write("$"+res+"PrecipDistribution\n")
    num_vals=len(frac)
    precip_dist_file.write(str(num_vals)+"\n")
    lines=""
    for i in range(num_vals):
        lines+=str(i+1)+" "+str(frac[i])+" "+str(abs(rad[i]))+"\n"
    precip_dist_file.write(lines)
    precip_dist_file.write("$End"+res+"PrecipDistribution")
    
def write_crss_file(values,name="simulation",res="Elset"):
    ####    Open crss file and write header
    #   set for isotropic input
    #
    print(f"opening file {name}.crss ...")
    precip_dist_file = open(name+".crss",'w')
    precip_dist_file.write("$"+res+"Crss\n")
    num_vals=len(values)
    precip_dist_file.write(str(num_vals)+" 1\n")
    #
    print(f"writing to file {name}.crss ...")
    for i in range(num_vals):
        precip_dist_file.write(str(i+1)+" "+str(values[i])+"\n")
    precip_dist_file.write("$End"+res+"Crss")
    #
    print(f"wrote file {name}.crss ...")
