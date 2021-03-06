# The functions here can be certainly optimised for speed using numpy better and no loops.

import numpy as np
import sys
import healpy as hp
from astropy.io import fits
import time
import matplotlib


import matplotlib.pyplot as plt

from deepmass import lens_data as ld
from deepmass import wiener


def power_function(k, sigmasignal = 17.5, amplitude = 1e6 ):
    """
    simple power spectrum example function
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
        return array * scaling + shift
    else:
        if clip==False:
            return array * scaling + shift
        else:
            return np.clip(array * scaling + shift, 0., 1.)
        

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


def mask_images(image_array, correct_mask):
    """
    Apply mask
    :param image_array: input image
    :param correct_mask: 1 for unmasked (visible pixels), 0 for mask
    :return: masked image
    """

    for i in range(len(image_array[:,0,0,0])):
        image_array[i,:,:,0] = image_array[i,:,:,0]*correct_mask

    return image_array


def radial_profile(data):
    """
    Compute the radial profile of 2d image
    :param data: 2d image
    :return: radial profile
    """
    center = data.shape[0]/2
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center)**2 + (y - center)**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile / data.shape[0]**2


def compute_spectrum_map(power1d,size):
    """
    takes 1D power spectrum and makes it an isotropic 2D map
    :param power: 1d power spectrum
    :param size:
    :return:
    """

    power_map = np.zeros((size, size), dtype = float)
    k_map =  np.zeros((size, size), dtype = float)

    for (i,j), val in np.ndenumerate(power_map):

        k1 = i - size/2.0
        k2 = j - size/2.0
        k_map[i, j] = (np.sqrt(k1*k1 + k2*k2))

        if k_map[i,j] == 0:
            power_map[i, j] = 1e-15
        else:
            power_map[i, j] = power1d[int(k_map[i, j])]

    return power_map


def generate_sv_maps(healpix_fits_file, data_file, output_base, n_outputs, power, Ncov,
                     size=256, mask=None, fast_noise=True, sigma_eps=0.2865, wiener_iter=30, reso=4.5):
    """
    Very hard coded function to make training data from healpix maps and save them

    :param healpix_fits_file: fits file location of true kappa healpix map
    :param data_file: catalogue of galaxies to match properties
    :param output_base: output location and file base
    :param n_outputs: number of patch realisations
    :param power: power spectrum for Wiener
    :param Ncov: Noise covariance diagonal for Wiener
    :param size: map size
    :param mask: healpix mask (optional)
    :param fast_noise: boolean. fast_noise==True means gaussian shape noise
    :param sigma_eps: default 0.25 for stdev of ellipticity
    :param wiener_iter: number of iterations (default 30)
    :param reso: resolution of projected map
    """


    kappa_map = hp.read_map(healpix_fits_file)
    nside = hp.npix2nside(len(kappa_map))

    if mask is None:
        print('mask calc')
        mask = np.zeros(hp.nside2npix(nside))
        th, ph = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
        ph[np.where(ph > np.pi)[0]] -= 2 * np.pi
        mask[np.where((th < np.pi * 0.5) & (ph > 0) & (ph < np.pi * 0.5))[0]] = 1.

    kappa_map = np.where(mask > 0.5, kappa_map, hp.UNSEEN)

    A_ft_diagonal = ld.ks_fourier_matrix(size)

    # read in data
    hdu_data = fits.open(data_file)
    data_cat = hdu_data[1].data

    if fast_noise==True:
        pixels = hp.ang2pix(nside, theta=0.5 * np.pi - np.deg2rad(data_cat.field('dec_gal')),
                            phi=np.deg2rad(data_cat.field('ra_gal')))
        count_map = np.bincount(pixels, minlength=hp.nside2npix(nside))

        count_gnomview = hp.gnomview(count_map, rot=[+75.0, -52.5], title='Mask',
                                     min=0, max=50, reso=reso, xsize=size, ysize=size,
                                     flip='geo', return_projected_map=True)
        _ = plt.close()

        std_map = np.where(count_gnomview > 0, sigma_eps / np.sqrt(count_gnomview), 0.)

    output_kappa_array = np.empty((n_outputs, size, size, 1), dtype=np.float32)
    output_ks_array = np.empty((n_outputs, size, size, 1), dtype=np.float32)

    if wiener_iter>0:
        output_wiener_array = np.empty((n_outputs, size, size, 1), dtype=np.float32)

    t = time.time()
    for i in range(n_outputs):

        if i % 10 == 0:
            print(str(i))



        patch = ld.random_map(kappa_map)

        # Randomly transpose signal with probability 0.5
        if np.random.randint(2) == 1:
            patch = patch.T

        shear = ld.ks_inv(patch, A_ft_diagonal)

        if fast_noise == True:
            e1_noise_map = np.random.normal(0.0, std_map)
            e2_noise_map = np.random.normal(0.0, std_map)

            e1_noisy = np.where(std_map != 0., e1_noise_map + shear.real, 0.)
            e2_noisy = np.where(std_map != 0., e2_noise_map + shear.imag, 0.)


        else:
            e1_des_noise, e2_des_noise = ld.shape_noise_realisation(data_cat.field('ra_gal'),
                                                                    data_cat.field('dec_gal'),
                                                                    data_cat.field('e1_gal_sens'),
                                                                    data_cat.field('e2_gal_sens'),
                                                                    nside)

            e1_noise_map = hp.gnomview(e1_des_noise, rot=[+75.0, -52.5], title='Mask',
                                       reso=reso, xsize=size, ysize=size, flip='geo',
                                       return_projected_map=True)
            _ = plt.close()

            e2_noise_map = hp.gnomview(e2_des_noise, rot=[+75.0, -52.5], title='Mask',
                                       reso=reso, xsize=size, ysize=size, flip='geo',
                                       return_projected_map=True)
            _ = plt.close()

            e1_noisy = np.where(e1_noise_map.mask == False,
                                e1_noise_map + shear.real, 0.)
            e2_noisy = np.where(e2_noise_map.mask == False,
                                e2_noise_map + shear.imag, 0.)

        output_kappa_array[i, :, :, 0] = np.real(np.array(patch))
        output_ks_array[i, :, :, 0] = np.real(ld.ks(e1_noisy + 1j * e2_noisy, A_ft_diagonal))

        if wiener_iter>0:
            output_wiener_array[i, :, :, 0], _ = wiener.filtering(e1_noisy, e2_noisy,
                                                                     power, Ncov, n_iter=wiener_iter)

    print('\n' + 'saving outputs \n')
    np.save(str(output_base + '_kappa_true'), output_kappa_array)
    np.save(str(output_base + '_KS'), output_ks_array)

    if wiener_iter>0:
        np.save(str(output_base + '_wiener'), output_wiener_array)


def mean_square_error(y_true, y_pred):
    """
    Calculate mean square error
    :param y_true: true data
    :param y_pred: predicted data
    :return: mean square error
    """
    return np.mean((y_pred.flatten() - y_true.flatten())**2.)
