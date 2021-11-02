import sys
import numpy as np
import time
import sys, os

import gc
import scipy.ndimage as ndimage


from deepmass import map_functions as mf
from deepmass import lens_data as ld
from deepmass import wiener
from deepmass import cnn_keras as cnn


import tensorflow as tf
print(tf.config.list_physical_devices('GPU'))


def augmented_foreground_generation(directory_synthesis, n_step, E_map_std, B_map_std, map_size):
    print(E_map_std, B_map_std)

    files = [filename for filename in os.listdir(directory_synthesis) if '.npy' in filename]
    nfiles= len(files)
    print(nfiles, flush=True)
    
    temp_map = np.load(directory_synthesis + files[0])
    input_foreground_array = np.empty((nfiles,temp_map.shape[0],temp_map.shape[0],2))

    qu_matrix = ld.ks_fourier_matrix(temp_map.shape[0])
    for i in range(nfiles):
        temp_map = np.load(directory_synthesis + files[i])

        temp_map = ld.ks(temp_map-np.mean(temp_map), qu_matrix) # convert QU to EB with periodic 512 maps

        input_foreground_array[i,:,:,0] = ndimage.gaussian_filter(temp_map.real, 1)
        input_foreground_array[i,:,:,1] = ndimage.gaussian_filter(temp_map.imag, 1)

    input_foreground_array[:,:,:,0] *= (E_map_std/np.std(input_foreground_array[:,:,:,0]))
    input_foreground_array[:,:,:,1] *= (B_map_std/np.std(input_foreground_array[:,:,:,1]))
    
    

    rotated_foreground_array  = np.vstack([tf.image.rot90(input_foreground_array, k = i) for i in range(4)])

    rotated_foreground_array = np.concatenate([rotated_foreground_array,
                                               tf.image.transpose(rotated_foreground_array)]).astype(np.float32)
    gc.collect()

    grid_step = np.meshgrid(np.linspace(0,map_size*2,n_step+1, dtype=int)[:-1],
                            np.linspace(0,map_size*2,n_step+1, dtype=int)[:-1])
    grid_i_index = grid_step[0].flatten()
    grid_j_index = grid_step[1].flatten()

    noisy_array = np.empty((len(grid_i_index)*rotated_foreground_array.shape[0], map_size,map_size,2), dtype=np.float32)
    print(noisy_array.shape, flush=True)

    for i in range(len(grid_i_index)):
        low_lim = i*rotated_foreground_array.shape[0]
        high_lim = (i+1)*rotated_foreground_array.shape[0]

        noisy_array[low_lim:high_lim] = np.roll(np.roll(rotated_foreground_array, shift=grid_i_index[i],axis=1),
                                                shift=grid_j_index[i],axis=2)[:,:int(map_size),:int(map_size),:]
        
    return noisy_array.astype(np.float32)
