#!/bin/sh

#SBATCH -t 12:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J stopping_treshold
#SBATCH -o stopping_treshold_%j.out
#SBATCH -e stopping_treshold_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20
#SBATCH --exclusive

# modules
module purge

# run the test
singularity run \
  --bind results:/project/results \
  container.sif \
  experiments/stopping-threshold.R
