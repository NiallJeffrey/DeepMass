
import numpy as np
import healpy as hp
import random
import matplotlib.pyplot as plt

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


def KS_fourier_matrix:

def KS(shear, fourier_forward_matrix):


def KS_inv(kappa, fourier_forward_matrix):