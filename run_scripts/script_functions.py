
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import sys

sys.path = ['../'] + sys.path

import scipy.ndimage as ndimage

import numpy as np
import time
import os


def load_data(file_list, verbose=True):
    """
    Reads in training data file and combines them into a single array
    :param file_list:
    :param verbose:
    :return: array of loaded data
    """

    if verbose:
        print('Loading ' + str(file_list[0]))

    data_array = np.load(file_list[0])

    for i in range(len(file_list) - 1):

        if verbose:
            print('Loading ' + str(file_list[i+1]))

        data_array = np.concatenate([data_array,
                                            np.load(file_list[i+1])])


    return data_array


