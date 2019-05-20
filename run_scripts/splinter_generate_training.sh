#!/bin/bash
#PBS -S /bin/bash
#PBS -q compute
#PBS -N array_test
#PBS -l nodes=1:ppn=1
#PBS -l walltime=100:00:00


module load dev_tools/oct2016/python-Anaconda-3-4.2.0

cd /share/splinter/ucapnje/DeepMass/run_scripts

## this jobfile should be called as a job array: e.g. qsub -t 0-58
PARTNAME=`printf "%02d" ${PBS_ARRAYID}`

cd /home/ucapnje/job_array_example

python generate_training.py $PARTNAME