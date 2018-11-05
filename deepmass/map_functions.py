# The functions here can be certainly optimised for speed using numpy better and no loops.

import numpy as np


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