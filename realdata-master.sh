#!/bin/sh

#SBATCH -N 6
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH -t 16:00:00
#
#SBATCH -J jobFarm
#SBATCH -o res_realdata_%j.out
#SBATCH -e res_realdata_%j.out
cat $0

declare -a data_sets=(\
    "news20" \
    "e2006-log1p-train" \
    "rcv1-train" \
    "e2006-tfidf-train" \
    "madelon-train" \
    "YearPredictionMSD-train" \
    "arcene" \
    "bc_tcga" \
    "colon-cancer" \
    "duke-breast-cancer" \
    "ijcnn1-train" \
    "scheetz" \
)

declare -a solvers=(\
    "hessian" \
    "working" \
    "celer" \
    "blitz" \
)

i=1

for data_set in ${data_sets[@]}
do
    for solver in ${solvers[@]}
    do
    srun -Q --exclusive --overlap -n 1 -N 1 \
        realdata-worker.sh $i $data_set $solver &> worker_${SLURM_JOB_ID}_${i}.out &
    i=$((i+1))
    sleep 1
    done
done

wait
