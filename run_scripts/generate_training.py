
import numpy as np
import sys
import os

from deepmass import map_functions as mf
from deepmass import lens_data as ld
from deepmass import wiener


command_line_index = sys.argv[1]

# kappa healpix directory
kappa_dir = '/share/splinter/ucapnje/picola_for_peaks/l-picola_peak_statistics/kappaA/kappa_nicaea_rescaled/'

list_kappa = os.listdir(kappa_dir)

print(len(list_kappa))

print(list_kappa[command_line_index])

print(kappa_dir + str(list_kappa[command_line_index]))