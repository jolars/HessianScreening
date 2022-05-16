#!/bin/sh

#SBATCH -t 12:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J path_length
#SBATCH -o path_length_%j.out
#SBATCH -e path_length_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20
#SBATCH --exclusive

# modules
module purge

# run the test
singularity run --bind results:/project/results container.sif \
  experiments/path-length.R
