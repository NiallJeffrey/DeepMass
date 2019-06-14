#!/bin/bash

# if getting permission denied errors do "chmod u+x program_name.sh" for this script

source activate tensorflow

pwd

cd share/DeepMass/run_scripts/

python test_wiener_unet.py


