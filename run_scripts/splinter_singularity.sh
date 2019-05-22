#!/bin/bash
#PBS -S /bin/bash
#PBS -q gpu
#PBS -N training_network
#PBS -l nodes=1:ppn=32
#PBS -l walltime=100:00:00


cd /share/splinter/ucapnje/DeepMass/run_scripts

singularity exec --nv -B /share/splinter/ucapnje/DeepMass:/home/ucapnje/DeepMass /share/data1/eme/ubuntu_tf_keras.img  ./DeepMass/run_scripts/submit_training.sh

#source activate tensorflow

#cd share/DeepMass/run_scripts/

#python simple_denoising.py


