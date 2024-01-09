#!/bin/bash 
#SBATCH -J job_name
#SBATCH -e error.%A 
#SBATCH -o output.%A 
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=48
mpirun -np 48 fepx
exit 0

#!/bin/bash
#
# Create tessellation from FF data
# Use file centvol, which contains the following for each grain:
# centroid_x centroid_y centroid_z volume
#
# neper -T \
#     -n 193 \
#     -domain "cube(1.3, 0.5, 0.5)" \
#     -morpho "centroidsize:file(centvols)" \
#     -reg 1 \
#     -o simulation
# #
# # Create a mesh
# #
# neper -M simulation.tess \
#     -order 2 \
#     -part 48 \
#     -o simulation
#
# Visualizations
# Straight tessellation
neper -V simulation.tess \
    -print simulation_tess

# Tessellation with mesh
neper -V simulation.msh \
    -dataelsetcol id \
    -print simulation_mesh
# Spheres of equivalent volume, requires:
# - A tessellation of just the domain
# - A file with just the centroids (centroids)
# - A file of the radii of the spheres of equivalent volume to the grains (radii)
neper -T -n 1 -domain "cube(1.3, 0.5, 0.5)" -o domain
neper -V domain.tess,centroids \
    -datacelltrs 1.3 \
    -datapointrad "file(radii)" \
    -print simulation_spheres
