
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from skimage.transform import resize
from deepmass import map_functions as mf
from deepmass import cnn_keras as cnn

import scipy.ndimage as ndimage
from keras.callbacks import TensorBoard
from keras.models import load_model

import numpy as np
import time
import os


print(os.getcwd())

resize_bool=False
map_size = 256
plot_results = True
output_dir = 'picola_script_outputs'
output_model_file = 'encoder_270318.h5'
n_epoch = 10
batch_size = 50
learning_rate_ks = 1e-4
learning_rate_wiener = 1e-4  # roughly 10-5 for 5 conv layers or 10-4 for 4 conv layers without bottleneck

sigma_smooth = 1.0

# rescaling quantities
scale_ks = 1.3
scale_wiener = 3.4

# make SV mask

print('loading mask \n')
counts = np.load('mice_mock/sv_counts.npy')

counts_shaped =  counts.reshape(map_size, int(counts.shape[0]/map_size),
                                map_size, int(counts.shape[1]/map_size)).sum(axis=1).sum(axis=2)
mask = np.float32(np.real(np.where(counts_shaped>0.0, 1.0, 0.0)))


# Load the data

print('loading data:')

print('- loading clean training')
train_array_clean = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_kappa_true.npy')
train_array_clean1 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output1_kappa_true.npy')
train_array_clean2 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output2_kappa_true.npy')
train_array_clean3 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output3_kappa_true.npy')
train_array_clean = np.concatenate([train_array_clean,train_array_clean1,train_array_clean2,train_array_clean3])

train_array_clean = ndimage.gaussian_filter(train_array_clean, sigma=(0,sigma_smooth,sigma_smooth, 0))

print('- loading ks training')
train_array_noisy = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_KS.npy')
train_array_noisy1 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output1_KS.npy')
train_array_noisy2 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output2_KS.npy')
train_array_noisy3 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output3_KS.npy')
train_array_noisy = np.concatenate([train_array_noisy,train_array_noisy1,train_array_noisy2,train_array_noisy3])

train_array_noisy = ndimage.gaussian_filter(train_array_noisy, sigma=(0,sigma_smooth*0.5,sigma_smooth*0.5, 0))


print('- loading wiener training')
train_array_wiener = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output_wiener.npy')
train_array_wiener1 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output1_wiener.npy')
train_array_wiener2 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output2_wiener.npy')
train_array_wiener3 = np.load(str(os.getcwd()) + '/picola_training/test_outputs/output3_wiener.npy')
train_array_wiener = np.concatenate([train_array_wiener,train_array_wiener1,train_array_wiener2,train_array_wiener3])

if resize_bool==True:
    train_array_clean = mf.downscale_images(train_array_clean, map_size, mask)
    train_array_noisy = mf.downscale_images(train_array_noisy, map_size, mask)
    train_array_wiener = mf.downscale_images(train_array_wiener, map_size, mask)
else:
    train_array_clean = mf.mask_images(train_array_clean, map_size, mask)
    train_array_noisy = mf.mask_images(train_array_noisy, map_size, mask)
    train_array_wiener = mf.mask_images(train_array_wiener, map_size, mask)


# remove maps where numerical errors give really low numbers (seem to occasionally happen - need to look into this)

x = np.where(np.sum(train_array_noisy[:,:,:,:] , axis = (1,2,3)) < -1e20)
mask_bad_data = np.ones(train_array_noisy[:,0,0,0].shape,dtype=np.bool)
mask_bad_data[x] = False


train_array_clean=train_array_clean[mask_bad_data,:,:,:] - np.mean(train_array_clean[mask_bad_data,:,:,:].flatten())
train_array_noisy=train_array_noisy[mask_bad_data,:,:,:]- np.mean(train_array_noisy[mask_bad_data,:,:,:].flatten())
train_array_wiener=train_array_wiener[mask_bad_data,:,:,:]- np.mean(train_array_wiener[mask_bad_data,:,:,:].flatten())

print(np.sum(train_array_clean), np.sum(train_array_noisy), np.sum(train_array_wiener))
print(np.max(np.abs(train_array_clean.flatten())), np.max(np.abs(train_array_noisy.flatten())), np.max(np.abs(train_array_wiener)))


# split a validation set 

test_array_clean = train_array_clean[:1000]
train_array_clean = train_array_clean[1000:]

test_array_noisy = train_array_noisy[:1000]
train_array_noisy = train_array_noisy[1000:]

test_array_wiener = train_array_wiener[:1000]
train_array_wiener = train_array_wiener[1000:]


# fraction of data out of 0 and 1 range
print('total pixels training = ' + str(len(train_array_clean.flatten())))
print('pixels out of range (truth with wiener scale) = ' + \
str(len(np.where(np.abs(-0.5 + mf.rescale_map(train_array_clean[:, :, :, 0], scale_wiener, 0.5).flatten()) > 0.5)[0])))
print('pixels out of range (wiener with wiener scale) = ' + \
str(len(np.where(np.abs(-0.5 + mf.rescale_map(train_array_wiener[:, :, :, 0], scale_wiener, 0.5).flatten()) > 0.5)[0])))
print('pixels out of range (truth with ks scale) = ' + \
str(len(np.where(np.abs(-0.5 + mf.rescale_map(train_array_clean[:, :, :, 0], scale_ks, 0.5).flatten()) > 0.5)[0])))
print('pixels out of range (ks with ks scale) = ' + \
str(len(np.where(np.abs(-0.5 + mf.rescale_map(train_array_noisy[:, :, :, 0], scale_ks, 0.5).flatten()) > 0.5)[0])))



# plot data
if plot_results:
    print('plotting data \n')
    n = 6  # how many images displayed
    plt.figure(figsize=(20, 15))
    for i in range(n):
        # display original
        plt.subplot(3, n, i + 1)
        plt.imshow(mf.rescale_map(train_array_clean[i, :, :, 0], scale_wiener, 0.5), origin='lower', cmap='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + n)
        plt.imshow(mf.rescale_map(train_array_noisy[i, :, :, 0], scale_ks, 0.5), origin='lower', cmap='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

        plt.subplot(3, n, i + 1 + 2*n)
        plt.imshow(mf.rescale_map(test_array_wiener[i, :, :, 0], scale_wiener, 0.5), origin='lower', cmap='inferno')
        plt.axis('off'), plt.colorbar(fraction=0.046, pad=0.04)

    plt.savefig(str(output_dir) + '/picola_data.pdf'), plt.close()


# Load encoder and train

print('training network KS \n')

autoencoder_instance = cnn.simple_model(map_size = map_size, learning_rate=learning_rate_ks)
autoencoder = autoencoder_instance.model()


print(n_epoch, batch_size, learning_rate_ks)

history_ks = cnn.LossHistory()

autoencoder.fit(mf.rescale_map(train_array_noisy, scale_ks, 0.5),
                mf.rescale_map(train_array_clean, scale_ks, 0.5),
                epochs=n_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(mf.rescale_map(test_array_noisy, scale_ks, 0.5),
                                 mf.rescale_map(test_array_clean, scale_ks, 0.5)),
                callbacks=[history_ks])

#print(history_ks.losses)
# save network
autoencoder.save(str(output_dir) + '/' + str(output_model_file))
#autoencoder = load_model(str(output_dir) + '/' + str(output_model_file))

# Load encoder and train

print('training network wiener \n')

autoencoder_instance_wiener = cnn.simple_model(map_size = map_size, learning_rate=learning_rate_wiener)
autoencoder_wiener = autoencoder_instance_wiener.model()

print(n_epoch, batch_size, learning_rate_wiener)

history_wiener = cnn.LossHistory()

autoencoder_wiener.fit(mf.rescale_map(train_array_wiener, scale_wiener, 0.5),
                mf.rescale_map(train_array_clean, scale_wiener, 0.5),
                epochs=n_epoch,
                batch_size=batch_size,
                shuffle=True,
                validation_data=(mf.rescale_map(test_array_wiener, scale_wiener, 0.5),
                                 mf.rescale_map(test_array_clean, scale_wiener, 0.5)),
                callbacks=[history_wiener])

#print(history_ks.losses)

# save network
autoencoder_wiener.save(str(output_dir) + '/wiener_' + str(output_model_file))
#autoencoder_wiener = load_model(str(output_dir) + '/wiener_' + str(output_model_file))

# plot result

if plot_results:
    print('plotting result \n')
    n_images = 6

    test_output = autoencoder.predict(mf.rescale_map(test_array_noisy[:n_images, :, :, :], scale_ks, 0.5))
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
    plt.savefig(str(output_dir) + '/picola_output.pdf'), plt.close()


# plot result

if plot_results:
    print('plotting result wiener \n')
    n_images = 6

    test_output = autoencoder_wiener.predict(mf.rescale_map(test_array_wiener[:n_images, :, :, :], scale_wiener, 0.5))
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
    plt.savefig(str(output_dir) + '/picola_output_wiener.pdf'), plt.close()

    n_images = 6

    test_output = autoencoder_wiener.predict(mf.rescale_map(test_array_wiener[:n_images, :, :, :], scale_wiener, 0.5))
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
    plt.savefig(str(output_dir) + '/picola_output_wiener2.pdf'), plt.close()
