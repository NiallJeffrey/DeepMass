

import matplotlib

matplotlib.use('agg')
import sys
import random

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

learning_rates = [1e-5, 3e-6]

temp_location = '/state/partition1/ucapnje/'

# make SV mask
print('loading mask \n')
mask = np.float32(np.real(np.where(np.load('../picola_training/Ncov.npy') > 1.0, 0.0, 1.0)))
print(mask.shape)


print('loading noisy data', flush=True)
t=time.time()
noisy_files = list(np.genfromtxt('data_file_lists/wiener_data_files_nongauss_noise.txt', dtype='str'))
noisy_files = [str(os.getcwd()) + s for s in noisy_files]
train_array_noisy = script_functions.load_data_preallocate(list(noisy_files[:]))

print(time.time()-t)
time.sleep(int(time.time()-t)/10)

# set masked regions to zero
print('\nApply mask', flush=True)
train_array_noisy = mf.mask_images(train_array_noisy, mask)

# remove maps where numerical errors give really high absolute values (seem to occasionally happen - need to look into this)
where_too_big = np.where(np.abs(np.sum(train_array_noisy[:, :, :, :], axis=(1, 2, 3))) > 1e10)

mask_bad_data = np.ones(train_array_noisy[:, 0, 0, 0].shape, dtype=np.bool)
print('\nNumber of bad files = ' + str(len(where_too_big[0])) + '\n')
mask_bad_data[where_too_big] = False

train_array_noisy = train_array_noisy[mask_bad_data, :, :, :]

print('\nShuffle and take fraction of test data')
random_indices = np.arange(len(train_array_noisy[:, 0, 0, 0]))
random.shuffle(random_indices)

train_array_noisy = train_array_noisy[random_indices]

print('Number of pixels sample = ' + str(len(train_array_noisy[:4000].flatten())))

print('pixels out of range (input/noisy) = ' +
      str(len(np.where(np.abs(-0.5 + mf.rescale_map(train_array_noisy[:4000], rescale_factor, 0.5).flatten()) > 0.5)[0])))

print('noisy array bytes = ' + str(train_array_noisy.nbytes))

train_array_noisy = mf.rescale_map(train_array_noisy, rescale_factor, 0.5, clip=True)

# split a validation set
test_array_noisy = train_array_noisy[:n_test]
train_array_noisy = train_array_noisy[n_test:]

print('Saving test array', flush=True)
t= time.time()
np.save(temp_location + 'temp_test_array_noisy', test_array_noisy)
print(time.time() - t)

print('Saving train array', flush=True)
t= time.time()
np.save(temp_location + 'temp_train_array_noisy', train_array_noisy)
print(time.time() - t)

test_array_noisy = None
train_array_noisy = None

# Load the clean data
print('loading data:', flush = True)
t=time.time()
clean_files = list(np.genfromtxt('data_file_lists/clean_data_files_nongauss_noise.txt', dtype='str'))
clean_files = [str(os.getcwd()) + s for s in clean_files]
train_array_clean = script_functions.load_data_preallocate(list(clean_files[:]))
#train_array_clean2 = script_functions.load_data(list(clean_files[:5]))
#print('Array equal = ' + str(np.array_equal(train_array_clean, train_array_clean2)))

print(time.time()-t)
time.sleep(int(time.time()-t)/10)

# set masked regions to zero
print('\nApply mask', flush=True)
train_array_clean = mf.mask_images(train_array_clean, mask)

# remove maps where numerical errors give really high absolute values (seem to occasionally happen - need to look into this)
train_array_clean = train_array_clean[mask_bad_data, :, :, :]

print('\nShuffle and take fraction of test data')
train_array_clean = train_array_clean[random_indices]

print('Number of pixels sample = ' + str(len(train_array_clean[:4000].flatten())))
print('pixels out of range (clean) = ' + \
      str(len(np.where(np.abs(-0.5 + mf.rescale_map(train_array_clean[:4000], rescale_factor, 0.5).flatten()) > 0.5)[0])))

print('clean array bytes = ' + str(train_array_clean.nbytes))

train_array_clean = mf.rescale_map(train_array_clean, rescale_factor, 0.5, clip=True)

# split a validation set
test_array_clean = train_array_clean[:n_test]
train_array_clean = train_array_clean[n_test:]


test_array_noisy = np.load(temp_location + 'temp_test_array_noisy.npy')
train_array_noisy = np.load(temp_location + 'temp_train_array_noisy.npy')

os.remove(temp_location + 'temp_test_array_noisy.npy')
os.remove(temp_location + 'temp_train_array_noisy.npy')

print('Test loss = ' + str(mf.mean_square_error(test_array_clean.flatten(), test_array_noisy.flatten())))
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

# Load encoder and train
print('training network wiener (no dropout) \n')


for learning_rate in learning_rates:
    print('\nsimple lr = ' + str(learning_rate))
    cnn_instance = cnn.simple_model(map_size=map_size, learning_rate=learning_rate)
    cnn_wiener = cnn_instance.model()

    print(n_epoch, batch_size, learning_rate)

    history = cnn.LossHistory()

    cnn_wiener.fit_generator(generator=train_gen,
                         epochs=n_epoch,
                         steps_per_epoch=np.ceil(train_array_noisy.shape[0] / 32),
                         validation_data=test_gen,
                         validation_steps=np.ceil(test_array_noisy.shape[0] / 32),
                         callbacks=[history], verbose=2)
    
    print('sleep')
    time.sleep(10)

    np.savetxt('losses_cnn_simple_' + str(learning_rate) + '.txt', history.losses)

    # save network
    cnn_wiener.save(str(h5_output_dir) + '/losses_cnn_simple_' + str(learning_rate) + '.h5')

    history = None
    test_output = cnn_wiener.predict(test_array_noisy)

    print('Test loss = ' + str(mf.mean_square_error(test_array_clean.flatten(), test_array_noisy.flatten())))
    print('Test pearson = ' + str(pearsonr(test_array_clean.flatten(), test_array_noisy.flatten())))


    print('Result loss = ' + str(mf.mean_square_error(test_array_clean.flatten(), test_output.flatten())))
    print('Result pearson = ' + str(pearsonr(test_array_clean.flatten(), test_output.flatten())))

    test_output = None

for learning_rate in learning_rates:
    print('\nunet deep lr = ' + str(learning_rate))
    cnn_instance = cnn.unet_simple_deep(map_size=map_size, learning_rate=learning_rate)
    cnn_wiener = cnn_instance.model()

    print(n_epoch, batch_size, learning_rate)

    history = cnn.LossHistory()

    cnn_wiener.fit_generator(generator=train_gen,
                         epochs=n_epoch,
                         steps_per_epoch=np.ceil(train_array_noisy.shape[0] / 32),
                         validation_data=test_gen,
                         validation_steps=np.ceil(test_array_noisy.shape[0] / 32),
                         callbacks=[history], verbose=2)

    print('sleep')
    time.sleep(10)

    np.savetxt('losses_unet_deep_' + str(learning_rate) + '.txt', history.losses)
    history = None
    # save network
    cnn_wiener.save(str(h5_output_dir) + '/losses_cnn_simple_' + str(learning_rate) + '.h5')

    test_output = cnn_wiener.predict(test_array_noisy)

    print('Test loss = ' + str(mf.mean_square_error(test_array_clean.flatten(), test_array_noisy.flatten())))
    print('Test pearson = ' + str(pearsonr(test_array_clean.flatten(), test_array_noisy.flatten())))

    print('Result loss = ' + str(mf.mean_square_error(test_array_clean.flatten(), test_output.flatten())))
    print('Result pearson = ' + str(pearsonr(test_array_clean.flatten(), test_output.flatten())))

    test_output = None
