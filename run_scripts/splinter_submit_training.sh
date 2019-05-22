#!/bin/bash
#PBS -S /bin/bash
#PBS -q gpu
#PBS -N training_network
#PBS -l nodes=1:ppn=32
#PBS -l walltime=100:00:00

singularity shell --nv -B /share/splinter/ucapnje:/home/ucapnje/share /share/data1/eme/ubuntu_tf_keras.img

source activate tensorflow

cd share/Deepmass/run_scripts

python simple_denoising


