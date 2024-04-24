#!/bin/sh

#SBATCH -t 08:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J simulateddata_extra
#SBATCH -o simulateddata_extra_%j.out
#SBATCH -e simulateddata_extra_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20
#SBATCH --exclusive

# modules
module purge

# run the test
singularity run --bind results:/project/results container.sif \
  experiments/simulateddata-extra.R
