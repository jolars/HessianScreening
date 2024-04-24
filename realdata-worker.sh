#!/bin/bash

cat $0

# receive my worker number
export WRK_NB=$1
export data_set=$2
export solver=$3

# # create worker-private subdirectory in $SNIC_TMP
export WRK_DIR=$SNIC_TMP/WRK_${WRK_NB}
mkdir $WRK_DIR

# # create a variable to address the "job directory"
export JOB_DIR=$SLURM_SUBMIT_DIR/job_${WRK_NB}

# # now copy the input data and program from there
cp -p ${SLURM_SUBMIT_DIR}/container.sif $WRK_DIR

# # change to the execution directory
cd $WRK_DIR

mkdir -p results/realdata
# # run the program

singularity run --no-home --bind results:/project/results/ container.sif \
  experiments/realdata.R $data_set $solver

# rescue the results back to job directory

cp -pr results/realdata ${SLURM_SUBMIT_DIR}/results/realdata

# clean up the local disk and remove the worker-private directory

cd $SNIC_TMP

rm -rf WRK_${WRK_NB}
