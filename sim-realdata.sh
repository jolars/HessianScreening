#!/bin/sh

#SBATCH -t 24:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J realdata
#SBATCH -o realdata_%j.out
#SBATCH -e realdata_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20

# modules
module purge

# bind results folders in Project to host
export SINGULARITY_BIND="results:/Project/results"

# run the test
singularity run container.sif realdata.R
