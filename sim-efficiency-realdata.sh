#!/bin/sh

#SBATCH -t 24:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J efficiency_realdata
#SBATCH -o efficiency_realdata_%j.out
#SBATCH -e efficiency_realdata_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20

# modules
module purge

singularity run --bind results:/Project/results container.sif efficiency-realdata.R
