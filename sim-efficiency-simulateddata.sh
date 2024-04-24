#!/bin/sh

#SBATCH -t 12:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J efficiency_simulateddata
#SBATCH -o efficiency_simulateddata_%j.out
#SBATCH -e efficiency_simulateddata_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20

# modules
module purge

singularity run \
  --bind results:/project/results \
  container.sif \
  experiments/efficiency-simulateddata.R
