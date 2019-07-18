

import matplotlib

matplotlib.use('agg')
import sys
import random
import gc

sys.path = ['../'] + sys.path

from deepmass import map_functions as mf
from deepmass import cnn_keras as cnn

import numpy as np
import time
import os
from scipy.stats import pearsonr

import script_functions

print(os.getcwd())

rescale_factor = 3.0

map_size = 256
n_test = int(8000)
plot_results = True
plot_output_dir = '../outputs/picola_script_outputs'
h5_output_dir = '../outputs/h5_files'
n_epoch = 20
batch_size = 32

input_location = '/home/ucapnje/testde_mount/ucapnje/training_data/wiener/'


# make SV mask
print('loading mask \n')
mask = np.float32(np.real(np.where(np.load('../picola_training/Ncov.npy') > 1.0, 0.0, 1.0)))
print(mask.shape)

print('Loading input data', flush=True)
t = time.time()
test_array_noisy = np.load(input_location + 'test_array_wiener.npy')
train_array_noisy = np.load(input_location + 'train_array_wiener.npy')
print(time.time()-t, flush=True)

gc.collect()

print('Loading clean data', flush=True)
t = time.time()
test_array_clean = np.load(input_location + 'test_array_clean.npy')
train_array_clean = np.load(input_location + 'train_array_clean.npy')
print(time.time()-t, flush=True)

gc.collect()

print('Test loss = ' + str(mf.mean_square_error(test_array_clean.flatten()/np.var(test_array_clean.flatten()), 
                                                test_array_noisy.flatten()/np.var(test_array_clean.flatten()))))
print('Test pearson = ' + str(pearsonr(test_array_clean.flatten(), test_array_noisy.flatten())), flush=True)



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


learning_rates=[1e-5,3e-6]

for learning_rate in learning_rates:

    print('unet simplest deeeper lr = ' + str(learning_rate))
    cnn_instance = cnn.unet_simplest_deeper2(map_size=map_size, learning_rate=learning_rate)
    cnn_wiener = cnn_instance.model()

    print(n_epoch, batch_size, learning_rate)

    history = cnn.LossHistory()

    cnn_wiener.fit_generator(generator=train_gen,
                         epochs=n_epoch,
                         steps_per_epoch=np.ceil(train_array_noisy.shape[0] / 32),
                         validation_data=test_gen,
                         validation_steps=np.ceil(test_array_noisy.shape[0] / 32),
                         callbacks=[history], verbose=2)

    print('Saving losses', flush=True)
#    np.savetxt('losses_unet_deep_' + str(learning_rate) + '.txt', history.losses)
    history = None

    # save network
    print('Save network', flush=True)
    cnn_wiener.save(str(h5_output_dir) + '/model_unet_simplest2' + str(learning_rate) + '.h5')

    test_output = cnn_wiener.predict(test_array_noisy)

    print('Test loss weights= ' + str(mf.mean_square_error(test_array_clean.flatten()/np.var(test_array_clean.flatten()), 
                                                           test_array_noisy.flatten()/np.var(test_array_clean.flatten()))))
    print('Test pearson = ' + str(pearsonr(test_array_clean.flatten(), test_array_noisy.flatten())))

    print('Result loss weights=' + str(mf.mean_square_error(test_array_clean.flatten()/np.var(test_array_clean.flatten()),
                                                            test_output.flatten()/np.var(test_array_clean.flatten()))))
    print('Result pearson = ' + str(pearsonr(test_array_clean.flatten(), test_output.flatten())), flush=True)

    test_output = None

    collected = gc.collect()
    print('Garbage collect: ' + str(collected), flush=True)


"""
for learning_rate in learning_rates:

    print('unet deeeper lr = ' + str(learning_rate))
    cnn_instance = cnn.unet_simple_deeper(map_size=map_size, learning_rate=learning_rate)
    cnn_wiener = cnn_instance.model()

    print(n_epoch, batch_size, learning_rate)

    history = cnn.LossHistory()

    cnn_wiener.fit_generator(generator=train_gen,
                         epochs=n_epoch,
                         steps_per_epoch=np.ceil(train_array_noisy.shape[0] / 32),
                         validation_data=test_gen,
                         validation_steps=np.ceil(test_array_noisy.shape[0] / 32),
                         callbacks=[history], verbose=2)

    print('Saving losses', flush=True)
#    np.savetxt('losses_unet_deep_' + str(learning_rate) + '.txt', history.losses)
    history = None

    # save network
    print('Save network', flush=True)
    cnn_wiener.save(str(h5_output_dir) + '/model_unet_simple' + str(learning_rate) + '.h5')

    test_output = cnn_wiener.predict(test_array_noisy)

    print('Test loss = ' + str(mf.mean_square_error(test_array_clean.flatten(), test_array_noisy.flatten())))
    print('Test pearson = ' + str(pearsonr(test_array_clean.flatten(), test_array_noisy.flatten())))

    print('Result loss = ' + str(mf.mean_square_error(test_array_clean.flatten(), test_output.flatten())))
    print('Result pearson = ' + str(pearsonr(test_array_clean.flatten(), test_output.flatten())), flush=True)

    test_output = None

    collected = gc.collect()
    print('Garbage collect: ' + str(collected), flush=True)

"""
