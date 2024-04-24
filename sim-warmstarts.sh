#!/bin/sh

#SBATCH -t 04:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J warmstarts
#SBATCH -o warmstarts_%j.out
#SBATCH -e warmstarts_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20

# modules
module purge

singularity run --bind results:/project/results container.sif\
  experiments/warm-starts.R
