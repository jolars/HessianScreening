#!/bin/sh

#SBATCH -t 72:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J ws_screening_contrib
#SBATCH -o ws_screening_contrib_%j.out
#SBATCH -e ws_screening_contrib_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20
#SBATCH --exclusive

# modules
module purge

# run the test
singularity run \
  --bind results:/Project/results \
  container.sif \
  ws-screening-contrib.R
