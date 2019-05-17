
import numpy as np
import healpy as hp
import random
import matplotlib.pyplot as plt
from scipy import fftpack

def random_map(masked_kappa, reso =4.5, size = 256, verbose=False):
    """
    This takes a masked healpix map and returns a projected patch
    """

    while 1== 1:
        proposal = hp.gnomview(masked_kappa,
                               rot=[np.random.uniform(0, 90.), np.random.uniform(0, 90.), np.random.uniform(0, 180)],
                               title='example', reso=reso, xsize=size, ysize=size,
                               flip='geo', return_projected_map=True)
        plt.close()
        #         print(proposal.shape)
        if len(np.where((proposal.mask).flatten() == True)[0]) > 1:
            if verbose==True:
                print('boundary error')
        else:
            if verbose==True:
                print('boundaries good')
            break

    return proposal.data


def make_healpix_map(ra, dec, weights, nside):
    """
    takes in a catalogue and makes masked healpix map of weights(e.g. kappa, sheae)
    """
    pixels = hp.ang2pix(nside, theta=0.5 * np.pi - np.deg2rad(dec), phi=np.deg2rad(ra))
    bincount = np.bincount(pixels, minlength=hp.nside2npix(nside))
    bincount_weighted = np.bincount(pixels, minlength=hp.nside2npix(nside), weights=weights)
    return np.where(bincount > 0.5, bincount_weighted / bincount, hp.UNSEEN)


def shape_noise_realisation(ra, dec, e1_orig, e2_orig, nside):
    """
    Takes a shape catalogue and returns healpix map with reordered shear values
    """

    gamma1_shuffle = np.copy(e1_orig)
    gamma2_shuffle = np.copy(e2_orig)

    random.shuffle(gamma1_shuffle)
    random.shuffle(gamma2_shuffle)
    e1_noise = make_healpix_map(ra, dec, gamma1_shuffle, nside)
    e2_noise = make_healpix_map(ra, dec, gamma2_shuffle, nside)
    return e1_noise, e2_noise


def ks_fourier_matrix(size):
    """

    :param size: size of square image
    :return: diagonal of forward (kappa to shear) Fourier operator
    """

    # k_modes = fftpack.fftfreq(size)
    # k1_grid = np.dstack(np.meshgrid(k_modes, k_modes))[:, :, 0]
    # k2_grid = k1_grid.T
    #
    # k1_vector = np.reshape(k1_grid, -1)
    # k2_vector = np.reshape(k2_grid, -1)
    #
    # k_squared = k1_vector * k1_vector + k2_vector * k2_vector
    # k_squared2 = np.where(k_squared == 0, 1e-18, k_squared)

    k_modes = fftpack.fftfreq(size)
    k1_grid = np.dstack(np.meshgrid(k_modes, k_modes))[:, :, 0]
    k1_vector = np.reshape(k1_grid, -1)
    k2_vector = np.reshape(k1_grid.T, -1)

    k_squared = k1_vector * k1_vector + k2_vector * k2_vector
    k_squared2 = np.where(k_squared == 0, 1e-18, k_squared)

    A_ft_diagonal = (k1_vector ** 2 - k2_vector ** 2 + 1j * 2.0 * k1_vector * k2_vector) / k_squared2
    return np.where(k_squared != 0.0, A_ft_diagonal, 1.0)



def ks(shear_map, fourier_forward_matrix=None):
    """
    Kaiser squires 1993
    :param shear_map: complex square shear map (e1 + i e2)
    :param fourier_forward_matrix: if matrix precalculated use it for speed
    :return: kappa map
    """

    if fourier_forward_matrix is None:
        fourier_forward_matrix = ks_fourier_matrix(shear_map.shape[0])

    fourier_shear_vector = np.fft.fft2(shear_map).flatten()

    return np.fft.ifft2(np.reshape(fourier_shear_vector/fourier_forward_matrix,shear_map.shape))


def ks_inv(kappa_map, fourier_forward_matrix=None):
    """
    Inverse Kaiser squires (kappa to shear)
    :param kappa: kappa map
    :param fourier_forward_matrix: if matrix precalculated use it for speed
    :return: complex shear map
    """
    if fourier_forward_matrix is None:
        fourier_forward_matrix = ks_fourier_matrix(kappa_map.shape[0])

    fourier_kappa_vector = np.fft.fft2(kappa_map).flatten()

    return np.fft.ifft2(np.reshape(fourier_kappa_vector*fourier_forward_matrix,kappa_map.shape))


