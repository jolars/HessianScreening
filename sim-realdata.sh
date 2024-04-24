#!/bin/sh

#SBATCH -t 36:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J realdata
#SBATCH -o realdata_%j.out
#SBATCH -e realdata_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20
#SBATCH --exclusive

# modules
module purge

singularity run --bind results:/project/results container.sif \
  experiments/realdata.R
