#!/bin/bash
#PBS -S /bin/bash
#PBS -q compute
#PBS -N training_data
#PBS -o /share/splinter/ucapnje/DeepMass/run_scripts/bash_outputs/training_data
#PBS -l nodes=1:ppn=1
#PBS -l walltime=100:00:00


module load dev_tools/oct2016/python-Anaconda-3-4.2.0

cd /share/splinter/ucapnje/DeepMass/run_scripts

## this jobfile should be called as a job array: e.g. qsub -t 0-58
PARTNAME=`printf "%02d" ${PBS_ARRAYID}`


python generate_training.py $PARTNAME