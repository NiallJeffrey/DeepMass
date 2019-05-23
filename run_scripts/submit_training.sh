#!/bin/bash

# if getting permission denied errors do "chmod u+x program_name.sh" for this script

source activate tensorflow

echo 
pwd


cd share/DeepMass/run_scripts/

python simple_denoising.py


