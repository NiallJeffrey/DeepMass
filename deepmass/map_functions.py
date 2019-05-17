# The functions here can be certainly optimised for speed using numpy better and no loops.

import numpy as np
import sys


def power_function(k, sigmasignal = 17.5, amplitude = 1e6 ):
    """
    simple power spectrum function
    :param k:
    :param sigmasignal:
    :param amplitude:
    :return: power
    """
    function =  np.exp(-float(k ) *float(k ) /( 2.0 * sigmasignal *sigmasignal) ) /(np.sqrt( 2.0 *np.pi ) *(sigmasignal))
    if function <= 0.0:
        return 1e-19
    else:
        return amplitude* function


def make_map(size, sigmasignal=None):
    """
    Makes a map with given power spectrum and no (very small) imaginary parts
    :param size:
    :param sigmasignal:
    :return: map, 2d fourier power_map
    """
    power_map = np.zeros((size, size), dtype=float)
    k_map = np.zeros((size, size), dtype=float)

    for (i, j), val in np.ndenumerate(power_map):

        k1 = i - size / 2.0
        k2 = j - size / 2.0
        k_map[i, j] = (np.sqrt(k1 * k1 + k2 * k2))

        if k_map[i, j] == 0:
            power_map[i, j] = 1e-15
        else:
            if sigmasignal is None:
                power_map[i, j] = power_function(k_map[i, j])
            else:
                power_map[i, j] = power_function(k_map[i, j], sigmasignal)

    power_sample = np.random.normal(loc=0.0, scale=np.sqrt(0.5 * np.reshape(power_map, -1))) + \
                   1j * np.random.normal(loc=0.0, scale=np.sqrt(0.5 * np.reshape(power_map, -1)))
    power_sample = np.reshape(power_sample, (size, size))

    # enforce hermitian and iFFT to get map
    map_fourier = np.zeros((size, size), dtype=complex)
    for (i, j,), value in np.ndenumerate(map_fourier):
        n = i - int(size / 2)
        m = j - int(size / 2)

        if n == 0 and m == 0:
            map_fourier[i, j] = 1e-15

        elif i == 0 or j == 0:
            map_fourier[i, j] = np.sqrt(2) * power_sample[i, j]

        else:
            map_fourier[i, j] = power_sample[i, j]  #
            map_fourier[size - i, size - j] = np.conj(power_sample[i, j])

    return np.fft.ifft2(np.fft.fftshift(map_fourier)), power_map


def rescale_map(array, scaling, shift, invert=False, clip = False):
    """
    This rescales an array, usually do to make range between 0 and 1
    :param array: original array
    :param scaling: multiplicative factor to elements
    :param shift: additive shift to the elements
    :param invert: if invert is True, the inverse of the rescale_map function will be done
    :param clip: if invert is True, the rescaled map is clipped between 0 and 1
    :return: rescaled array
    """
    if invert is True:
        shift = -shift / scaling
        scaling = 1.0 / scaling
        return (np.copy(array) * scaling + shift)
    else:
        if clip==False:
            return (np.copy(array) * scaling + shift)
        else:
            return (np.clip(np.copy(array) * scaling + shift, 0., 1.))
        


def rescale_unit_test():
    """
    this checks the invert option works
    """
    original_array = np.linspace(0, 500, 5)
    new_array = rescale_map(rescale_map(original_array, 0.1, 0.5, False), 0.1, 0.5, True)

    if (original_array == new_array).all():
        print('Unit passed passed: rescale_map')
    else:
        print(original_array)
        print(new_array)
        print('Unit test failed: rescale_map')
        sys.exit()



def downscale_images(image_array, new_size, correct_mask):
    image_array_new = np.empty((len(image_array[:,0,0,0]), new_size, new_size, 1), dtype = np.float32)

    for i in range(len(image_array[:,0,0,0])):
        image_array_new[i,:,:,0] = resize(image_array[i,:,:,0], (new_size,new_size))*correct_mask

    return image_array_new


def mask_images(image_array, new_size, correct_mask):
    image_array_new = np.empty((len(image_array[:,0,0,0]), new_size, new_size, 1), dtype = np.float32)

    for i in range(len(image_array[:,0,0,0])):
        image_array_new[i,:,:,0] = image_array[i,:,:,0]*correct_mask

    return image_array_new





# # function to compute the power spectrum of a 2d image
def radial_profile(data):
    center = data.shape[0]/2
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center)**2 + (y - center)**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile / data.shape[0]**2


def compute_spectrum_map(Px,size):
    """
    takes 1D power spectrum and makes it an isotropic 2D map
    """
    power_map = np.zeros((size, size), dtype = float)
    k_map =  np.zeros((size, size), dtype = float)

    for (i,j), val in np.ndenumerate(power_map):

        k1 = i - size/2.0
        k2 = j - size/2.0
        k_map[i, j] = (np.sqrt(k1*k1 + k2*k2))

        if k_map[i,j]==0:
            #print(i,j)
            power_map[i, j] = 1e-15
        else:
            #print(k_map[i, j])
            power_map[i, j] = Px[int(k_map[i, j])]
    return power_map

