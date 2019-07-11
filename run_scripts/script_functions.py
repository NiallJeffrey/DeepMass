
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
import sys
import random

sys.path = ['../'] + sys.path


from deepmass import map_functions as mf
from deepmass import cnn_keras as cnn

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

    data_array = np.array(np.load(file_list[0]), dtype='float32')

    for i in range(len(file_list) - 1):

        if verbose:
            print('Loading ' + str(file_list[i+1]))

        data_array = np.concatenate([data_array,
                                            np.array(np.load(file_list[i+1]), dtype='float32')])
        data_array = np.array(data_array,dtype = 'float32')

    return np.array(data_array, dtype='float32')



def load_data_list(file_list):
    """

    :param file_list: list of data files
    :return:
    """

    data_array = np.concatenate([np.array(np.load(f), dtype = 'float32') for f in file_list])

    return data_array


def load_data_preallocate(file_list, n_images_per_file=None):
    """
    This is the prefered function to load. It is much faster and memory efficient as it generates
    an array of the correct size at the start
    :param file_list: list of float32 data files
    :param n_images_per_file: number of images in each file of the file list in order
    :return: data array
    """

    n_images_per_file = np.array(n_images_per_file, dtype='int')
    first_file = np.array(np.load(file_list[0]), dtype='float32')
    print('loaded first file: ' + str(file_list[0]), flush=True)

    shape_tuple = first_file.shape

    if n_images_per_file is None:
        new_shape = (shape_tuple[0]*len(file_list), shape_tuple[1], shape_tuple[2], shape_tuple[3])

        data_array = np.empty(new_shape, dtype = 'float32')

        data_array[:shape_tuple[0]] = first_file

        for i in range(len(file_list) - 1):

            print('Loading ' + str(file_list[i+1]), flush=True)

            data_array[(shape_tuple[0]*(i+1)):(shape_tuple[0]*(i+2))] = np.load(file_list[i+1])

        return data_array

    else:
        new_shape = (np.sum(n_images_per_file), shape_tuple[1], shape_tuple[2], shape_tuple[3])
        data_array = np.empty(new_shape, dtype='float32')

        data_array[:n_images_per_file[0]] = first_file[:n_images_per_file[0]]

        position_counter = n_images_per_file[0]

        for i in range(len(file_list) - 1):
            i+=1
            print('Loading ' + str(file_list[i]), flush=True)

            data_array[position_counter:(position_counter+n_images_per_file[i])] = \
                np.load(file_list[i])[:n_images_per_file[i]]

            position_counter += n_images_per_file[i]

        return data_array



def plot_cnn(clean, noisy, reconstructed, output_file, vmin=0.3, vmax=0.7):
    """
    plots the clean maps, noisy maps and reconstruction
    :param clean:
    :param noisy:
    :param reconstructed:
    :param vmin: minimum colour on imshow
    :param vmax: maximum colour on imshow
    :param output_file: saves to a file
    """

    n = 6  # how many images displayed
    n_images = len(clean[:, 0, 0, 0])
    random_image_index = random.randint(0, n_images - 6)
    plt.figure(figsize=(20, 15))
    for i in range(n):
        # display original
        plt.subplot(3, n , i + 1)
        plt.imshow(clean[i+ random_image_index, :, :, 0],
                   origin='lower', cmap='inferno', clim=(vmin,vmax))
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(noisy[i+ random_image_index, :, :, 0],
                   origin='lower', cmap='inferno', clim=(vmin,vmax))
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(reconstructed[i+ random_image_index, :, :, 0],
                   origin='lower', cmap='inferno', clim=(vmin,vmax))
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig(output_file), plt.close()


def plot_noisy_clean(clean, noisy, output_file, vmin=0.3, vmax=0.7):
    """
    plots the clean maps, noisy maps and reconstruction
    :param clean:
    :param noisy:
    :param reconstructed:
    :param vmin: minimum colour on imshow
    :param vmax: maximum colour on imshow
    :param output_file: saves to a file
    """

    n = 6  # how many images displayed
    n_images = len(clean[:, 0, 0, 0])
    random_image_index = random.randint(0, n_images - 6)
    plt.figure(figsize=(20, 11))
    for i in range(n):
        # display original
        plt.subplot(2, n , i + 1)
        plt.imshow(clean[i+ random_image_index, :, :, 0],
                   origin='lower', cmap='inferno', clim=(vmin,vmax))
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(2, n, i + 1 + n)
        plt.imshow(noisy[i+ random_image_index, :, :, 0],
                   origin='lower', cmap='inferno', clim=(vmin,vmax))
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig(output_file), plt.close()
