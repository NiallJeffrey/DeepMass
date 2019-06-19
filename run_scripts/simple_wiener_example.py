
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
n_test = int(2000)
plot_results = True
plot_output_dir = '../outputs/picola_script_outputs'
h5_output_dir = '../outputs/h5_files'
output_model_file = '210519.h5'
n_epoch = 20
batch_size = 32
learning_rate_ks = 3e-5

# rescaling quantities
scale_ks = 1.

# make SV mask
print('loading mask \n')
mask = np.float32(np.real(np.where(np.load('../picola_training/Ncov.npy')>1.0, 0.0, 1.0)))
print(mask.shape)

# Load the data
print('loading data:')
clean_files = list(np.genfromtxt('data_file_lists/clean_data_files.txt', dtype ='str'))
clean_files = [str(os.getcwd()) + s for s in clean_files]
train_array_clean = script_functions.load_data(list(clean_files[15:30]))

noisy_files = list(np.genfromtxt('data_file_lists/wiener_data_files.txt', dtype ='str'))
noisy_files = [str(os.getcwd()) + s for s in noisy_files]
train_array_noisy = script_functions.load_data(list(noisy_files[15:30]))

# set masked regions to zero
print('\nApply mask')
train_array_clean = mf.mask_images(train_array_clean, mask)
train_array_noisy = mf.mask_images(train_array_noisy, mask)

# remove maps where numerical errors give really low numbers (seem to occasionally happen - need to look into this)
x = np.where(np.sum(np.abs(train_array_noisy[:,:,:,:]), axis = (1,2,3)) > 1e18)
mask_bad_data = np.ones(train_array_noisy[:,0,0,0].shape,dtype=np.bool)
print('\nNumber of bad files = ' + str(len(x[0])) + '\n')
mask_bad_data[x] = False

train_array_clean=train_array_clean[mask_bad_data,:,:,:]
train_array_noisy=train_array_noisy[mask_bad_data,:,:,:]


print('\nShuffle and take fraction of test data')
random_indices = np.arange(len(train_array_clean[:, 0, 0, 0]))
random.shuffle(random_indices)
train_array_clean=train_array_clean[random_indices]
train_array_noisy=train_array_noisy[random_indices]

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



#Load encoder and train
print('training network \n')

cnn_instance = cnn.simple_model(map_size = map_size, learning_rate=learning_rate_ks)
cnn_ks = cnn_instance.model()

print(n_epoch, batch_size, learning_rate_ks)

history_ks = cnn.LossHistory()

print(train_gen)
print(train_array_noisy.shape)
print(test_array_clean.shape)
print(train_array_noisy.shape[0] // 32)

cnn_ks.fit_generator(generator=train_gen,
                      epochs=n_epoch,
                     steps_per_epoch= np.ceil(train_array_noisy.shape[0] / 32),
                     validation_data=test_gen,
                      validation_steps=np.ceil(test_array_noisy.shape[0] / 32),
                      use_multiprocessing=True,
                     callbacks=[history_ks], verbose=2)

# save network
cnn_ks.save(str(h5_output_dir) + '/' + str(output_model_file))


# plot result
if plot_results:

    print('Plotting results. Saving to: ' + str(plot_output_dir) + '/picola_output.png')

    # Apply trained CNN to test data
    random_index = int(np.random.uniform(0,len(test_array_noisy[:, 0, 0, 0])-1000))
    test_output = cnn_ks.predict(test_array_noisy[random_index:(random_index+1000)])
    test_output = mf.rescale_map(test_output, scale_ks, 0.5, True)

    script_functions.plot_cnn(mf.rescale_map(test_array_clean[random_index:(random_index+1000)],
                                             scale_ks, 0.5, True),
                              mf.rescale_map(test_array_noisy[random_index:(random_index+1000)],
                                             scale_ks, 0.5, True),
                              test_output,
                              str(plot_output_dir) + '/picola_output.png',
                              -0.025,0.025)
