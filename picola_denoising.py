
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from skimage.transform import resize
from deepmass import map_functions as mf
from deepmass import cnn_keras as cnn

import scipy.ndimage as ndimage
from keras.callbacks import TensorBoard

import numpy as np
import time
import os

print(os.getcwd())

resize_bool=True
map_size = 128
plot_results = True
output_dir = 'picola_script_outputs'
output_model_file = 'encoder_200318b.h5'
n_epoch = 100
batch_size = 30
learning_rate = 1e-5

sigma_smooth = 1.

# rescaling quantities
scale_kappa = 8.0
scale_ks = 8.
scale_wiener = 8.0

# make SV mask

print('loading mask \n')
counts = np.load('mice_mock/sv_counts.npy')

counts_shaped =  counts.reshape(map_size, int(counts.shape[0]/map_size),
                                map_size, int(counts.shape[1]/map_size)).sum(axis=1).sum(axis=2)
mask = np.where(counts_shaped>0.0, 1.0, 0.0)
mask = np.float32(mask.real)


def downscale_images(image_array, new_size, correct_mask):
    image_array_new = np.empty((len(image_array[:,0,0,0]), new_size, new_size, 1), dtype = np.float32)

    for i in range(len(image_array[:,0,0,0])):
        image_array_new[i,:,:,0] = resize(image_array[i,:,:,0], (new_size,new_size))*correct_mask

    return image_array_new


# Load the data

print('loading data:')

print('- loading clean training')
train_array_clean = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_kappa_true.npy')
train_array_clean1 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output1_kappa_true.npy')
train_array_clean = np.concatenate([train_array_clean,train_array_clean1])

train_array_clean = ndimage.gaussian_filter(train_array_clean, sigma=(0,sigma_smooth,sigma_smooth, 0))

print('- loading ks training')
train_array_noisy = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_KS.npy')
train_array_noisy1 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output1_KS.npy')
train_array_noisy = np.concatenate([train_array_noisy,train_array_noisy1])

train_array_noisy = ndimage.gaussian_filter(train_array_noisy, sigma=(0,sigma_smooth*0.5,sigma_smooth*0.5, 0))


print('- loading wiener training')
train_array_wiener = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_wiener.npy')
train_array_wiener1 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output1_wiener.npy')
train_array_wiener = np.concatenate([train_array_wiener,train_array_wiener1])

if resize_bool==True:
    train_array_clean = downscale_images(train_array_clean, map_size, mask)
    train_array_noisy = downscale_images(train_array_noisy, map_size, mask)
    train_array_wiener = downscale_images(train_array_wiener, map_size, mask)

x = np.where(np.sum(train_array_noisy[:,:,:,:] , axis = (1,2,3)) < -1e20)
mask_bad_data = np.ones(train_array_noisy[:,0,0,0].shape,dtype=np.bool)
mask_bad_data[x] = False

train_array_clean=train_array_clean[mask_bad_data,:,:,:]
train_array_noisy=train_array_noisy[mask_bad_data,:,:,:]
train_array_wiener=train_array_wiener[mask_bad_data,:,:,:]
print(np.sum(train_array_clean), np.sum(train_array_noisy), np.sum(train_array_wiener))
print(np.max(np.abs(train_array_clean)), np.max(np.abs(train_array_noisy)), np.max(np.abs(train_array_wiener)))

print('- loading clean testing')
test_array_clean = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_kappa_true_test500.npy')
test_array_clean = ndimage.gaussian_filter(test_array_clean, sigma=(0,sigma_smooth, sigma_smooth, 0))
print('- loading noisy testing \n')
test_array_noisy = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_KS_test500.npy')
test_array_noisy = ndimage.gaussian_filter(test_array_noisy, sigma=(0,sigma_smooth*0.5,sigma_smooth*0.5, 0))

print('- loading wiener testing \n')
test_array_wiener = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_wiener_test500.npy')


if resize_bool==True:
    test_array_clean = downscale_images(test_array_clean, map_size, mask)
    test_array_noisy = downscale_images(test_array_noisy, map_size, mask)
    test_array_wiener = downscale_images(test_array_wiener, map_size, mask)

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
        plt.imshow(mf.rescale_map(train_array_clean[i, :, :, 0], scale_kappa, 0.5), origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(mf.rescale_map(train_array_noisy[i, :, :, 0], scale_ks, 0.5), origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(mf.rescale_map(test_array_wiener[i, :, :, 0], scale_wiener, 0.5), origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig(str(output_dir) + '/picola_data.pdf'), plt.close()


# Load encoder and train

print('training network KS \n')

autoencoder_instance = cnn.autoencoder_model(map_size = map_size, learning_rate=learning_rate)
autoencoder = autoencoder_instance.model()

autoencoder.fit(mf.rescale_map(train_array_noisy, scale_ks, 0.5),
                mf.rescale_map(train_array_clean, scale_kappa, 0.5),
                epochs=n_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(mf.rescale_map(test_array_noisy, scale_ks, 0.5),
                                 mf.rescale_map(test_array_clean, scale_kappa, 0.5)),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


# save network
#autoencoder.save(str(output_dir) + '/' + str(output_model_file))

# Load encoder and train

print('training network wiener \n')

autoencoder_instance_wiener = cnn.simple_model_residual(map_size = map_size, learning_rate=learning_rate)
autoencoder_wiener = autoencoder_instance_wiener.model()

autoencoder_wiener.fit(mf.rescale_map(train_array_wiener, scale_wiener*3., 0.5),
                mf.rescale_map(train_array_clean, scale_kappa*3., 0.5),
                epochs=n_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(mf.rescale_map(test_array_wiener, scale_wiener*3., 0.5),
                                 mf.rescale_map(test_array_clean, scale_kappa*3., 0.5)),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])


# save network
autoencoder_wiener.save(str(output_dir) + '/wiener_' + str(output_model_file))

# plot result

if plot_results:
    print('plotting result \n')
    n_images = 6

    test_output = autoencoder.predict(mf.rescale_map(test_array_noisy[:n_images, :, :, :], scale_ks, 0.5))
#     test_output = mf.rescale_map(test_output, scale_ks, 0.5, True)


    plt.figure(figsize=(30, 15))
    for i in range(n_images):
        # display original
        plt.subplot(3, n_images, i + 1)
        plt.imshow(mf.rescale_map(test_array_clean[i, :, :, 0], scale_kappa*3., 0.5), origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + n_images)
        plt.imshow(mf.rescale_map(test_array_noisy[i, :, :, 0], scale_wiener*3., 0.5), origin='lower')
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

    test_output = autoencoder_wiener.predict(mf.rescale_map(test_array_wiener[:n_images, :, :, :], scale_wiener*3., 0.5))
#     test_output = mf.rescale_map(test_output, scale_wiener*3., 0.5, True)


    plt.figure(figsize=(30, 15))
    for i in range(n_images):
        # display original
        plt.subplot(3, n_images, i + 1)
        plt.imshow(mf.rescale_map(test_array_clean[i, :, :, 0], scale_kappa*3., 0.5), origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + n_images)
        plt.imshow(mf.rescale_map(test_array_wiener[i, :, :, 0], scale_wiener*3., 0.5), origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n_images, i + 1 + 2 * n_images)
        plt.imshow(test_output[i, :, :, 0], origin='lower')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.axis('off')
    plt.savefig(str(output_dir) + '/picola_output_wiener.pdf'), plt.close()
