
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
n_test = int(5000)
plot_results = True
plot_output_dir = '../outputs/picola_script_outputs'
h5_output_dir = '../outputs/h5_files'
n_epoch = 20
batch_size = 30
# learning_rate_ks = 1e-5
# learning_rate_wiener = 1e-5 # roughly 10-5 for 5 conv layers or 10-4 for 4 conv layers without bottleneck

sigma_smooth = 0.01

# rescaling quantities
scale_ks = 1.
scale_wiener = 2.5

# make SV mask

print('loading mask \n')
mask = np.float32(np.real(np.where(np.load('../picola_training/Ncov.npy')>1.0, 0.0, 1.0)))
print(mask.shape)

# Load the data

print('loading data:')
print('- loading clean training')
clean_files = list(np.genfromtxt('data_file_lists/clean_data_files.txt', dtype ='str'))
clean_files = [str(os.getcwd()) + s for s in clean_files]
train_array_clean = script_functions.load_data(list(clean_files[20:30]))
train_array_clean = ndimage.gaussian_filter(train_array_clean, sigma=(0,sigma_smooth,sigma_smooth, 0))

print('- loading ks training')
noisy_files = list(np.genfromtxt('data_file_lists/noisy_data_files.txt', dtype ='str'))
noisy_files = [str(os.getcwd()) + s for s in noisy_files]
train_array_noisy = script_functions.load_data(list(noisy_files[20:30]))
train_array_noisy = ndimage.gaussian_filter(train_array_noisy, sigma=(0,sigma_smooth,sigma_smooth, 0))


print('- loading wiener training')
wiener_files = list(np.genfromtxt('data_file_lists/wiener_data_files.txt', dtype ='str'))
wiener_files = [str(os.getcwd()) + s for s in wiener_files]
train_array_wiener = script_functions.load_data(list(wiener_files[20:30]))
train_array_wiener= ndimage.gaussian_filter(train_array_wiener, sigma=(0,sigma_smooth,sigma_smooth, 0))

# set masked regions to zero
print('\nApply mask')
train_array_clean = mf.mask_images(train_array_clean, mask)
train_array_noisy = mf.mask_images(train_array_noisy, mask)
train_array_wiener = mf.mask_images(train_array_wiener, mask)

# remove maps where numerical errors give really low numbers (seem to occasionally happen - need to look into this)
x = np.where(np.sum(np.abs(train_array_noisy[:,:,:,:]), axis = (1,2,3)) > 1e18)
mask_bad_data = np.ones(train_array_noisy[:,0,0,0].shape,dtype=np.bool)
print('\nNumber of bad files = ' + str(len(x[0])) + '\n')
mask_bad_data[x] = False


train_array_clean=train_array_clean[mask_bad_data,:,:,:]
train_array_noisy=train_array_noisy[mask_bad_data,:,:,:]
train_array_wiener=train_array_wiener[mask_bad_data,:,:,:]


print('\nShuffle and take fraction of test data')
# random order
random_indices = np.arange(len(train_array_clean[:, 0, 0, 0]))
random.shuffle(random_indices)
train_array_clean=train_array_clean[random_indices]
train_array_noisy=train_array_noisy[random_indices]
train_array_wiener = train_array_wiener[random_indices]



# split a validation set
test_array_clean = train_array_clean[:n_test]
train_array_clean = train_array_clean[n_test:]

test_array_noisy = train_array_noisy[:n_test]
train_array_noisy = train_array_noisy[n_test:]

test_array_wiener = train_array_wiener[:n_test]
train_array_wiener = train_array_wiener[n_test:]



# plot data
if plot_results:
    print('plotting data \n')
    n = 6  # how many images displayed
    n_images = len(train_array_clean[:, 0, 0, 0])
    random_image_index = random.randint(0, n_images - 6)
    plt.figure(figsize=(20, 15))
    for i in range(n):
        # display original
        plt.subplot(3, n , i + 1)
        plt.imshow(np.clip(mf.rescale_map(train_array_clean[i+ random_image_index, :, :, 0], scale_wiener, 0.5),
                           0.,1.),
                   origin='lower', cmap='inferno', clim=(0.3,0.7))
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(np.clip(mf.rescale_map(train_array_noisy[i + random_image_index, :, :, 0], scale_ks, 0.5),
                   0.0,1.0),
                   origin='lower', cmap='inferno', clim=(0.3,0.7))
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(np.clip(mf.rescale_map(train_array_wiener[i + random_image_index, :, :, 0], scale_wiener, 0.5),
                   0.,1.),
                   origin='lower', cmap='inferno', clim=(0.3,0.7))
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig(str(plot_output_dir) + '/picola_data.png'), plt.close()


# First do the unet network

print('\n Training unet network learning rate = 1e-5, no smoothing')

cnn_instance = cnn.unet_simple(map_size = map_size, learning_rate=1e-5)
cnn_ks = cnn_instance.model()
print(n_epoch, batch_size, 1e-5)

history_ks = cnn.LossHistory()

cnn_ks.fit(np.clip(mf.rescale_map(train_array_noisy, scale_ks, 0.5),0.,1.0),
           np.clip(mf.rescale_map(train_array_clean, scale_ks, 0.5), 0.0,1.0),
                epochs=n_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(np.clip(mf.rescale_map(test_array_noisy, scale_ks, 0.5),0.,1.),
                                np.clip(mf.rescale_map(test_array_clean, scale_ks, 0.5), 0., 1.)),
                callbacks=[history_ks], verbose=2)

cnn_ks.save(str(h5_output_dir) + '/unet_simple_KS_1e5')


if plot_results:
    print('plotting result \n')
    n_images = 6

    test_output = cnn_ks.predict(mf.rescale_map(test_array_noisy[:n_images, :, :, :], scale_ks, 0.5))
    test_output = mf.rescale_map(test_output, scale_ks, 0.5, True)


    plt.figure(figsize=(30, 15))
    for i in range(n_images):
        # display original
        plt.subplot(3, n_images, i + 1)
#        plt.imshow(mf.rescale_map(test_array_clean[i, :, :, 0], scale_ks, 0.5), origin='lower')
        plt.imshow(test_array_clean[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + n_images)
        #plt.imshow(mf.rescale_map(test_array_noisy[i, :, :, 0], scale_ks, 0.5), origin='lower')
        plt.imshow(test_array_noisy[i, :, :, 0], origin='lower', clim = (-0.1,0.1), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)


        plt.subplot(3, n_images, i + 1 + 2 * n_images)
        plt.imshow(test_output[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.axis('off')
    plt.savefig(str(plot_output_dir) + '/unet_simple_KS_1e5.png'), plt.close()



print('\n Training KS unet network learning rate = 1e-4')

cnn_instance = cnn.unet_simple(map_size = map_size, learning_rate=1e-4)
cnn_ks = cnn_instance.model()
print(n_epoch, batch_size, 1e-4)

history_ks = cnn.LossHistory()

cnn_ks.fit(np.clip(mf.rescale_map(train_array_noisy, scale_ks, 0.5),0.,1.0),
           np.clip(mf.rescale_map(train_array_clean, scale_ks, 0.5), 0.0,1.0),
                epochs=n_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(np.clip(mf.rescale_map(test_array_noisy, scale_ks, 0.5),0.,1.),
                                np.clip(mf.rescale_map(test_array_clean, scale_ks, 0.5), 0., 1.)),
                callbacks=[history_ks])

cnn_ks.save(str(h5_output_dir) + '/unet_simple_KS_1e4')


if plot_results:
    print('plotting result \n')
    n_images = 6

    test_output = cnn_ks.predict(mf.rescale_map(test_array_noisy[:n_images, :, :, :], scale_ks, 0.5))
    test_output = mf.rescale_map(test_output, scale_ks, 0.5, True)


    plt.figure(figsize=(30, 15))
    for i in range(n_images):
        # display original
        plt.subplot(3, n_images, i + 1)
#        plt.imshow(mf.rescale_map(test_array_clean[i, :, :, 0], scale_ks, 0.5), origin='lower')
        plt.imshow(test_array_clean[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + n_images)
        #plt.imshow(mf.rescale_map(test_array_noisy[i, :, :, 0], scale_ks, 0.5), origin='lower')
        plt.imshow(test_array_noisy[i, :, :, 0], origin='lower', clim = (-0.1,0.1), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)


        plt.subplot(3, n_images, i + 1 + 2 * n_images)
        plt.imshow(test_output[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.axis('off')
    plt.savefig(str(plot_output_dir) + '/unet_simple_KS_1e4.png'), plt.close()


print('\n Training Wiener unet network learning rate = 1e-5, no smoothing')

cnn_instance_wiener = cnn.unet_simple(map_size = map_size, learning_rate=1e-5)
cnn_wiener = cnn_instance_wiener.model()
print(n_epoch, batch_size, 1e-5)

history_wiener = cnn.LossHistory()

cnn_wiener.fit(np.clip(mf.rescale_map(train_array_wiener, scale_wiener, 0.5),0.,1.0),
               np.clip(mf.rescale_map(train_array_clean, scale_wiener, 0.5),0.,1.),
                epochs=n_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(np.clip(mf.rescale_map(test_array_wiener, scale_wiener, 0.5),0.,1.),
                                 np.clip(mf.rescale_map(test_array_clean, scale_wiener, 0.5),0.,1.)),
                callbacks=[history_wiener])


# save network
cnn_wiener.save(str(h5_output_dir) + '/unet_simpler_wiener_1e5')

# plot result
if plot_results:
    print('plotting result wiener \n')
    n_images = 6

    test_output = cnn_wiener.predict(mf.rescale_map(test_array_wiener[:n_images, :, :, :], scale_wiener, 0.5))
    test_output = mf.rescale_map(test_output, scale_wiener, 0.5, True)

    plt.figure(figsize=(30, 15))
    for i in range(n_images):
        # display original
        plt.subplot(3, n_images, i + 1)
        #plt.imshow(mf.rescale_map(test_array_clean[i, :, :, 0], scale_wiener, 0.5), origin='lower')
        plt.imshow(test_array_clean[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + n_images)
        #plt.imshow(mf.rescale_map(test_array_wiener[i, :, :, 0], scale_wiener, 0.5), origin='lower')
        plt.imshow(test_array_wiener[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + 2 * n_images)
        plt.imshow(test_output[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.axis('off')
    plt.savefig(str(plot_output_dir) + '/uent_simple_wiener_1e5.png'), plt.close()

    n_images = 6

    test_output = cnn_wiener.predict(mf.rescale_map(test_array_wiener[:n_images, :, :, :], scale_wiener, 0.5))
    test_output = mf.rescale_map(test_output, scale_wiener, 0.5, True)


    plt.figure(figsize=(30, 15))
    for i in range(n_images):
        # display original
        plt.subplot(3, n_images, i + 1)
        #plt.imshow(mf.rescale_map(test_array_clean[i, :, :, 0], scale_wiener, 0.5), origin='lower')
        plt.imshow(test_array_clean[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + n_images)
        #plt.imshow(mf.rescale_map(test_array_wiener[i, :, :, 0], scale_wiener, 0.5), origin='lower')
#        plt.imshow(test_array_wiener[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.imshow(test_array_noisy[i, :, :, 0], origin='lower', clim = (-0.1,0.1), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + 2 * n_images)
        plt.imshow(test_output[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.axis('off')
    plt.savefig(str(plot_output_dir) + '/unet_simple_wiener2_1e5.png'), plt.close()



print('\n Training Wiener unet network learning rate = 1e-4, no smoothing')

cnn_instance_wiener = cnn.unet_simple(map_size = map_size, learning_rate=1e-4)
cnn_wiener = cnn_instance_wiener.model()
print(n_epoch, batch_size, 1e-4)

history_wiener = cnn.LossHistory()

cnn_wiener.fit(np.clip(mf.rescale_map(train_array_wiener, scale_wiener, 0.5),0.,1.0),
               np.clip(mf.rescale_map(train_array_clean, scale_wiener, 0.5),0.,1.),
                epochs=n_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(np.clip(mf.rescale_map(test_array_wiener, scale_wiener, 0.5),0.,1.),
                                 np.clip(mf.rescale_map(test_array_clean, scale_wiener, 0.5),0.,1.)),
                callbacks=[history_wiener])


# save network
cnn_wiener.save(str(h5_output_dir) + '/unet_simple_wiener_1e4')

# plot result
if plot_results:
    print('plotting result wiener \n')
    n_images = 6

    test_output = cnn_wiener.predict(mf.rescale_map(test_array_wiener[:n_images, :, :, :], scale_wiener, 0.5))
    test_output = mf.rescale_map(test_output, scale_wiener, 0.5, True)

    plt.figure(figsize=(30, 15))
    for i in range(n_images):
        # display original
        plt.subplot(3, n_images, i + 1)
        #plt.imshow(mf.rescale_map(test_array_clean[i, :, :, 0], scale_wiener, 0.5), origin='lower')
        plt.imshow(test_array_clean[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + n_images)
        #plt.imshow(mf.rescale_map(test_array_wiener[i, :, :, 0], scale_wiener, 0.5), origin='lower')
        plt.imshow(test_array_wiener[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + 2 * n_images)
        plt.imshow(test_output[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.axis('off')
    plt.savefig(str(plot_output_dir) + '/unet_simple_wiener_1e4.png'), plt.close()

    n_images = 6

    test_output = cnn_wiener.predict(mf.rescale_map(test_array_wiener[:n_images, :, :, :], scale_wiener, 0.5))
    test_output = mf.rescale_map(test_output, scale_wiener, 0.5, True)


    plt.figure(figsize=(30, 15))
    for i in range(n_images):
        # display original
        plt.subplot(3, n_images, i + 1)
        #plt.imshow(mf.rescale_map(test_array_clean[i, :, :, 0], scale_wiener, 0.5), origin='lower')
        plt.imshow(test_array_clean[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + n_images)
        #plt.imshow(mf.rescale_map(test_array_wiener[i, :, :, 0], scale_wiener, 0.5), origin='lower')
#        plt.imshow(test_array_wiener[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.imshow(test_array_noisy[i, :, :, 0], origin='lower', clim = (-0.1,0.1), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + 2 * n_images)
        plt.imshow(test_output[i, :, :, 0], origin='lower', clim = (-0.02,0.02), cmap ='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.axis('off')
    plt.savefig(str(plot_output_dir) + '/unet_simple_wiener2_1e4.png'), plt.close()