
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from deepmass import map_functions as mf
from deepmass import cnn_keras as cnn

from keras.callbacks import TensorBoard

import numpy as np
import time

map_size = 256
max_training_data = 1000
plot_results = True
output_dir = 'example_script_outputs'
output_model_file = 'encoder_140318.h5'

# make SV mask

print('loading mask \n')
counts = np.load('mice_mock/sv_counts.npy')

counts_shaped =  counts.reshape(map_size, int(counts.shape[0]/map_size),
                                map_size, int(counts.shape[1]/map_size)).sum(axis=1).sum(axis=2)
mask = np.where(counts_shaped>0.0, 1.0, 0.0)
mask = np.float32(mask.real)

if plot_results:
    print('plotting mask \n')
    plt.figure()
    plt.imshow(mask, origin = 'lower'), plt.colorbar(fraction=0.046, pad=0.04)
    plt.savefig(str(output_dir) + '/example_mask.pdf'), plt.close()

# Load the data

print('loading data:')

print('- loading clean training')
train_array_clean = np.load('misc_data/train_clean_256_10000.npy')[:max_training_data]

print('- loading noisy training')
train_array_noisy = np.load('misc_data/train_noisy_256_10000.npy')[:max_training_data]

print('- loading clean testing')
test_array_clean = np.load('misc_data/test_clean_256_1000.npy')

print('- loading noisy testing \n')
test_array_noisy = np.load('misc_data/test_noisy_256_1000.npy')

# plot data
if plot_results:
    print('plotting data \n')
    n = 6  # how many images displayed
    plt.figure(figsize=(20, 10))
    for i in range(n):
        # display original
        plt.subplot(3, n, i + 1)
        plt.imshow(train_array_clean[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(train_array_noisy[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig(str(output_dir) + '/example_data.pdf'), plt.close()


# Load encoder and train

print('training network \n')

autoencoder_instance = cnn.autoencoder_model(map_size = 256)
autoencoder = autoencoder_instance.model()

autoencoder.fit(mf.rescale_map(train_array_noisy, 0.25, 0.5),
                mf.rescale_map(train_array_clean, 0.25, 0.5),
                epochs=1,
                batch_size=100,
                shuffle=True,
                validation_data=(mf.rescale_map(test_array_noisy, 0.25, 0.5),
                                 mf.rescale_map(test_array_clean, 0.25, 0.5)),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


# save network
autoencoder.save(str(output_dir) + '/' + str(output_model_file))

# plot result

if plot_results:
    print('plotting result \n')
    n_images = 6

    test_output = autoencoder.predict(mf.rescale_map(test_array_noisy[:n_images, :, :, :], 0.25, 0.5))
    test_output = mf.rescale_map(test_output, 0.25, 0.5, True)


    plt.figure(figsize=(30, 15))
    for i in range(n_images):
        # display original
        plt.subplot(3, n_images, i + 1)
        plt.imshow(test_array_clean[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + n_images)
        plt.imshow(test_array_noisy[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + 2 * n_images)
        plt.imshow(test_output[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.axis('off')
    plt.savefig(str(output_dir) + '/example_output.pdf'), plt.close()
