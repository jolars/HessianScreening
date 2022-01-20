#!/bin/sh

#SBATCH -t 72:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J warm_start_vs_no_warm_start
#SBATCH -o warm_start_vs_no_warm_start_%j.out
#SBATCH -e warm_start_vs_no_warm_start_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20
#SBATCH --exclusive

# modules
module purge

# run the test
singularity run \
  --bind results:/Project/results \
  container.sif \
  warm-start-vs-no-warm-start.R
