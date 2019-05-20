
import numpy as np
import sys
import os
import time

from deepmass import map_functions as mf

command_line_index = int(sys.argv[1])

# kappa healpix directory
kappa_dir = '/share/splinter/ucapnje/picola_for_peaks/l-picola_peak_statistics/kappaA/kappa_nicaea_rescaled/'

list_kappa = os.listdir(kappa_dir)

print(list_kappa[command_line_index])

picola_file = kappa_dir + str(list_kappa[command_line_index])

print(picola_file)

# Load power spectrum and covariance matrix
power_spectrum = np.load('../picola_training/temp_wiener_power_fiducial.py')
Ncov = np.load('../picola_training/Ncov.npy')

# make training data
output_file_base = '../picola_training/nicaea_rescaled/training_data' + str(command_line_index) + '/sv_training_'


t = time.time()
mf.generate_sv_maps(picola_file,
                    '../mice_mock/cat_DES_SV_zmean_final.fits',
                    output_file_base,
                    n_outputs = 40, power = power_spectrum, Ncov=Ncov,  sigma_eps=0.286)

print(time.time() -t)

