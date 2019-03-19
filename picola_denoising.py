
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt

from deepmass import map_functions as mf
from deepmass import cnn_keras as cnn

from keras.callbacks import TensorBoard

import numpy as np
import time
import os

print(os.getcwd())

map_size = 256
max_training_data = 7500
plot_results = True
output_dir = 'picola_script_outputs'
output_model_file = 'encoder_190318.h5'
n_epoch = 4

# rescaling quantities
scale_kappa = 2.0
scale_ks = 0.5
scale_wiener = 20.

# Load the data

print('loading data:')

print('- loading clean training')
train_array_clean = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_kappa_true.npy')

print('- loading ks training')
train_array_noisy = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_KS.npy')

print('- loading wiener training')
train_array_wiener = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_wiener.npy')

x = np.where(np.sum(train_array_noisy[:,:,:,:] , axis = (1,2,3)) < -1e20)
mask_bad_data = np.ones(train_array_noisy[:,0,0,0].shape,dtype=np.bool)
mask_bad_data[x] = False

train_array_clean=train_array_clean[mask_bad_data,:,:,:]
train_array_noisy=train_array_noisy[mask_bad_data,:,:,:]
train_array_wiener=train_array_wiener[mask_bad_data,:,:,:]

print('- loading clean testing')
test_array_clean = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_kappa_true_test500.npy')

print('- loading noisy testing \n')
test_array_noisy = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_KS_test500.npy')

print('- loading wiener testing \n')
test_array_wiener = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_wiener_test500.npy')


x = np.where(np.sum(test_array_noisy[:,:,:,:] , axis = (1,2,3)) < -1e20)
mask_bad_data = np.ones(test_array_noisy[:,0,0,0].shape,dtype=np.bool)
mask_bad_data[x] = False

test_array_clean=test_array_clean[mask_bad_data,:,:,:]
test_array_noisy=test_array_noisy[mask_bad_data,:,:,:]
test_array_wiener=test_array_wiener[mask_bad_data,:,:,:]

# plot data
if plot_results:
    print('plotting data \n')
    n = 6  # how many images displayed
    plt.figure(figsize=(20, 15))
    for i in range(n):
        # display original
        plt.subplot(3, n, i + 1)
        plt.imshow(train_array_clean[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(train_array_noisy[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(train_array_wiener[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig(str(output_dir) + '/picola_data.pdf'), plt.close()


# Load encoder and train

print('training network KS \n')

autoencoder_instance = cnn.autoencoder_model(map_size = map_size)
autoencoder = autoencoder_instance.model()

autoencoder.fit(mf.rescale_map(train_array_noisy, scale_ks, 0.5),
                mf.rescale_map(train_array_clean, scale_kappa, 0.5),
                epochs=n_epoch,
                batch_size=30,
                shuffle=True,
                validation_data=(mf.rescale_map(test_array_noisy, scale_ks, 0.5),
                                 mf.rescale_map(test_array_clean, scale_kappa, 0.5)),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


# save network
autoencoder.save(str(output_dir) + '/' + str(output_model_file))

# Load encoder and train

print('training network wiener \n')

autoencoder_instance_wiener = cnn.autoencoder_model(map_size = map_size)
autoencoder_wiener = autoencoder_instance_wiener.model()

autoencoder_wiener.fit(mf.rescale_map(train_array_wiener, scale_wiener, 0.5),
                mf.rescale_map(train_array_clean, scale_kappa, 0.5),
                epochs=n_epoch,
                batch_size=30,
                shuffle=True,
                validation_data=(mf.rescale_map(test_array_wiener, scale_wiener, 0.5),
                                 mf.rescale_map(test_array_clean, scale_kappa, 0.5)),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


# save network
autoencoder_wiener.save(str(output_dir) + '/wiener_' + str(output_model_file))

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
    plt.savefig(str(output_dir) + '/picola_output.pdf'), plt.close()


# plot result

if plot_results:
    print('plotting result wiener \n')
    n_images = 6

    test_output = autoencoder_wiener.predict(mf.rescale_map(test_array_wiener[:n_images, :, :, :], 0.25, 0.5))
    test_output = mf.rescale_map(test_output, 0.25, 0.5, True)


    plt.figure(figsize=(30, 15))
    for i in range(n_images):
        # display original
        plt.subplot(3, n_images, i + 1)
        plt.imshow(test_array_clean[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + n_images)
        plt.imshow(test_array_wiener[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + 2 * n_images)
        plt.imshow(test_output[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.axis('off')
    plt.savefig(str(output_dir) + '/picola_output_wiener.pdf'), plt.close()