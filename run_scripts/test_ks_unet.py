import matplotlib

matplotlib.use('agg')
import matplotlib.pyplot as plt
import sys
import random

sys.path = ['../'] + sys.path

from deepmass import map_functions as mf
from deepmass import cnn_keras as cnn

import scipy.ndimage as ndimage
from keras.callbacks import TensorBoard
from keras.models import load_model

import numpy as np
import time
import os

import script_functions

print(os.getcwd())

map_size = 256
n_test = int(4000)
plot_results = True
plot_output_dir = '../outputs/picola_script_outputs'
h5_output_dir = '../outputs/h5_files'
output_model_file = '210519.h5'
n_epoch = 10
batch_size = 32
learning_rates = [1e-4, 3e-5, 1e-5, 3e-6]

# rescaling quantities
scale_ks = 1.

# make SV mask
print('loading mask \n')
mask = np.float32(np.real(np.where(np.load('../picola_training/Ncov.npy') > 1.0, 0.0, 1.0)))
print(mask.shape)

# Load the data
print('loading data:')
clean_files = list(np.genfromtxt('data_file_lists/clean_data_files.txt', dtype='str'))
clean_files = [str(os.getcwd()) + s for s in clean_files]
train_array_clean = script_functions.load_data(list(clean_files[:]))

noisy_files = list(np.genfromtxt('data_file_lists/noisy_data_files.txt', dtype='str'))
noisy_files = [str(os.getcwd()) + s for s in noisy_files]
train_array_noisy = script_functions.load_data(list(noisy_files[:]))

# set masked regions to zero
print('\nApply mask')
train_array_clean = mf.mask_images(train_array_clean, mask)
train_array_noisy = mf.mask_images(train_array_noisy, mask)

# remove maps where numerical errors give really low numbers (seem to occasionally happen - need to look into this)
x = np.where(np.sum(np.abs(train_array_noisy[:, :, :, :]), axis=(1, 2, 3)) > 1e18)
mask_bad_data = np.ones(train_array_noisy[:, 0, 0, 0].shape, dtype=np.bool)
print('\nNumber of bad files = ' + str(len(x[0])) + '\n')
mask_bad_data[x] = False

train_array_clean = train_array_clean[mask_bad_data, :, :, :]
train_array_noisy = train_array_noisy[mask_bad_data, :, :, :]

print('\nShuffle and take fraction of test data')
random_indices = np.arange(len(train_array_clean[:, 0, 0, 0]))
random.shuffle(random_indices)
train_array_clean = train_array_clean[random_indices]
train_array_noisy = train_array_noisy[random_indices]

print('Number of pixels sample = ' + str(len(train_array_clean[:2000].flatten())))
print('pixels out of range (truth) = ' +
      str(len(np.where(np.abs(-0.5 + mf.rescale_map(train_array_clean[:2000], scale_ks, 0.5).flatten()) > 0.5)[0])))
print('pixels out of range (input/noisy) = ' + \
      str(len(np.where(np.abs(-0.5 + mf.rescale_map(train_array_noisy[:2000], scale_ks, 0.5).flatten()) > 0.5)[0])))

print('clean array bytes = ' + str(train_array_clean.nbytes))
print('noisy array bytes = ' + str(train_array_noisy.nbytes))

train_array_clean = mf.rescale_map(train_array_clean, scale_ks, 0.5, clip=True)
train_array_noisy = mf.rescale_map(train_array_noisy, scale_ks, 0.5, clip=True)

# split a validation set
test_array_clean = train_array_clean[:n_test]
train_array_clean = train_array_clean[n_test:]

test_array_noisy = train_array_noisy[:n_test]
train_array_noisy = train_array_noisy[n_test:]

print('Test loss = ' + str(mf.mean_square_error(test_array_clean.flatten(), test_array_noisy.flatten())))

if plot_results:
    print('Plotting data. Saving to: ' + str(plot_output_dir) + '/picola_data.png')
    script_functions.plot_noisy_clean(test_array_clean,
                                      test_array_noisy,
                                      output_file=str(plot_output_dir) + '/picola_data.png')



train_gen = cnn.batch_generator(train_array_noisy, train_array_clean, gen_batch_size=batch_size)
test_gen = cnn.batch_generator(test_array_noisy, test_array_clean, gen_batch_size=batch_size)

print(train_gen)
print(train_array_noisy.shape)
print(test_array_clean.shape)
print(train_array_noisy.shape[0] // 32)

# Load encoder and train
print('training network KS \n')

for learning_rate_ks in learning_rates:
    print('unet_simple lr = ' + str(learning_rate_ks))
    cnn_instance = cnn.unet_simple(map_size=map_size, learning_rate=learning_rate_ks)
    cnn_ks = cnn_instance.model()

    print(n_epoch, batch_size, learning_rate_ks)

    history_ks = cnn.LossHistory()

    cnn_ks.fit_generator(generator=train_gen,
                         epochs=n_epoch,
                         steps_per_epoch=np.ceil(train_array_noisy.shape[0] / 32),
                         validation_data=test_gen,
                         validation_steps=np.ceil(test_array_noisy.shape[0] / 32),
                         use_multiprocessing=True,
                         callbacks=[history_ks], verbose=2)

for learning_rate_ks in learning_rates:
    print('unet lr = ' + str(learning_rate_ks))
    cnn_instance = cnn.unet(map_size=map_size, learning_rate=learning_rate_ks)
    cnn_ks = cnn_instance.model()

    print(n_epoch, batch_size, learning_rate_ks)

    history_ks = cnn.LossHistory()


    cnn_ks.fit_generator(generator=train_gen,
                         epochs=n_epoch,
                         steps_per_epoch=np.ceil(train_array_noisy.shape[0] / 32),
                         validation_data=test_gen,
                         validation_steps=np.ceil(test_array_noisy.shape[0] / 32),
                         use_multiprocessing=True,
                         callbacks=[history_ks], verbose=2)



for learning_rate_ks in learning_rates:
    print('unet_simple_deep lr = ' + str(learning_rate_ks))
    cnn_instance = cnn.unet_simple_deep(map_size=map_size, learning_rate=learning_rate_ks)
    cnn_ks = cnn_instance.model()

    print(n_epoch, batch_size, learning_rate_ks)

    history_ks = cnn.LossHistory()

    cnn_ks.fit_generator(generator=train_gen,
                         epochs=n_epoch,
                         steps_per_epoch=np.ceil(train_array_noisy.shape[0] / 32),
                         validation_data=test_gen,
                         validation_steps=np.ceil(test_array_noisy.shape[0] / 32),
                         use_multiprocessing=True,
                         callbacks=[history_ks], verbose=2)
