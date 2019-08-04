# derived from Kostas' code

import numpy as np
# import lens_data as ld



# define the kappa to shear operator
def H_operator(ka_map, kb_map):
    # ka_map and kb_map should be of the same size
    [nx, ny] = ka_map.shape

    ka_map_fft = np.fft.fft2(ka_map)
    kb_map_fft = np.fft.fft2(kb_map)

    f1, f2 = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

    p1 = f1 * f1 - f2 * f2
    p2 = 2 * f1 * f2
    f2 = f1 * f1 + f2 * f2
    f2[0, 0] = 1  # avoid division with zero
    kafc = (p1 * ka_map_fft - p2 * kb_map_fft) / f2
    kbfc = (p1 * kb_map_fft + p2 * ka_map_fft) / f2

    g1_map = np.fft.ifft2(kafc).real
    g2_map = np.fft.ifft2(kbfc).real

    return g1_map, g2_map


# define the shear to convergence operator
def H_adjoint(g1_map, g2_map):
    [nx, ny] = g1_map.shape

    g1_map_ifft = np.fft.ifft2(g1_map)
    g2_map_ifft = np.fft.ifft2(g2_map)

    f1, f2 = np.meshgrid(np.fft.fftfreq(ny), np.fft.fftfreq(nx))

    p1 = f1 * f1 - f2 * f2
    p2 = 2 * f1 * f2
    f2 = f1 * f1 + f2 * f2
    f2[0, 0] = 1
    g1fc = (p1 * g1_map_ifft + p2 * g2_map_ifft) / f2
    g2fc = (p1 * g2_map_ifft - p2 * g1_map_ifft) / f2

    kappa1 = np.fft.fft2(g1fc).real
    kappa2 = np.fft.fft2(g2fc).real

    return kappa1, kappa2


def filtering(gamma1, gamma2, Px_map, Ncov, n_iter=500):

    nx = gamma1.shape[0]
    xg = np.zeros((nx, nx)) + 1j * np.zeros((nx, nx))

    # find the minimum noise variance
    tau = np.min(Ncov)

    # set the step size
    eta = 1.83 * tau

    # compute signal coefficient
    Esn = eta / Ncov

    # calculate the wiener filter coefficients
    Wfc = Px_map / (Px_map + eta)

    for n in range(n_iter):
        t1, t2 = H_operator(xg.real, xg.imag)  # H * xg
        t1, t2 = H_adjoint(Esn * (gamma1 - t1), Esn * (gamma2 - t2))  # H^T(eta / Sn * (y- H * xg))
        t = xg + (t1 + 1j * t2)  # xg + H^T(eta / Sn * (y- H * xg))

        tf = np.fft.fftshift(np.fft.fft2(t))
        xgf = Wfc * tf  # wiener filtering in fourier space
        xg = np.fft.ifft2(np.fft.fftshift(xgf))

    return xg.real, xg.imag


# def prox_wiener_new(shear, Px_map, Ncov, n_iter=500):
#     # initiallize
#     nx = shear.shape[0]
#     xg = np.zeros((nx, nx)) + 1j * np.zeros((nx, nx))
#     # find the minimum noise variance
#     tau = np.min(Ncov)
#
#     # set the step size
#     eta = 1.83 * tau
#
#     # compute signal coefficient
#     Esn = eta / Ncov
#
#     # calculate the wiener filter coefficients
#     Wfc = Px_map / (Px_map + eta)
#
#     for n in range(n_iter):
#         t = ld.ks_inv(xg)  # H * xg
#         t = ld.ks(Esn * (shear.real - t.real) + 1j * Esn * (shear.imag - t.imag))  # H^T(eta / Sn * (y- H * xg))
#         t = xg + t  # xg + H^T(eta / Sn * (y- H * xg))
#
#         tf = np.fft.fftshift(np.fft.fft2(t))
#         xgf = Wfc * tf  # wiener filtering in fourier space
#         xg = np.fft.ifft2(np.fft.fftshift(xgf))
#
#     return xg
