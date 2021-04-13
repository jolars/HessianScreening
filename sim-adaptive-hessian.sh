#!/bin/sh

#SBATCH -t 02:00:00

#SBATCH --mail-user=johan.larsson@stat.lu.se
#SBATCH --mail-type=ALL

#SBATCH -J adaptivehessian
#SBATCH -o adaptivehessian_%j.out
#SBATCH -e adaptivehessian_%j.err

#SBATCH -N 1
#SBATCH --tasks-per-node=20

# modules
module purge

singularity run --bind results:/Project/results container.sif adaptive-hessian.R
