#!/bin/sh

#SBATCH -t 24:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J simulateddata
#SBATCH -o simulateddata_%j.out
#SBATCH -e simulateddata_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20

# modules
module purge

# bind results folders in Project to host
export SINGULARITY_BIND="results:/Project/results"

# run the test
singularity run container.sif simulateddata.R
