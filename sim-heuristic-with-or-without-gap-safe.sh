#!/bin/sh

#SBATCH -t 12:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J heuristic_with_or_without_gap_safe
#SBATCH -o heuristic_with_or_without_gap_safe_%j.out
#SBATCH -e heuristic_with_or_without_gap_safe_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20
#SBATCH --exclusive

# modules
module purge

# run the test
singularity run \
  --bind results:/project/results \
  container.sif \
  experiments/heuristic-with-or-without-gap-safe.R
