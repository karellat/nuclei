import numpy as np
import os.path
from itertools import product
import scipy.io
from oct2py import octave, Struct
from glob import glob


def matlab_image_to_invariants(img,
                               srz,
                               types,
                               appell_rank,
                               appell_type,
                               appell_param,
                               appell_weight):
    if types > 0:
        v = 2 + (types % 2)
    else:
        v = 3

    coords = np.linspace(-1, 1, srz)
    [x, y, z] = np.meshgrid(coords, coords, coords)
    x = x.flatten(order='F')
    y = y.flatten(order='F')
    z = z.flatten(order='F')
    matlab_polynomials = matlab_appell_polynomials_recursive_3d(appell_rank, appell_rank, appell_rank,
                                                                x[..., np.newaxis], y[..., np.newaxis],
                                                                z[..., np.newaxis],
                                                                t=appell_type.value,
                                                                s=appell_param,
                                                                w=appell_weight.value)

    matlab_moments = matlab_appell_moments_3D_predef(img.astype(np.float64), matlab_polynomials,
                                                     appell_rank, appell_rank, appell_rank)


    moments2invariants = np.load("rot3Dinv10indep.npz", allow_pickle=True)
    invariant_ind = list(moments2invariants['ind'][0][:77])
    invariant_coef = list(moments2invariants['coef'][0][:77])
    invariant_sizes = np.array([np.sum(ind[0, :]) / v for ind in invariant_ind])

    return matlab_cafmi3dst(invariant_coef,
                            invariant_ind,
                            matlab_moments,
                            types=types,
                            typeg=1)


def get_images():
    for file in glob('nucleus*npz'):
        yield np.load(file)['arr_0']


def matlab_pochham(x, k):
    octave.addpath('./matlab')
    return octave.pochham(x, k)


def matlab_readinv3dst(fname):
    octave.addpath('./matlab')
    return octave.readinv3dst(fname)


def matlab_gauss_hermite_polynoms(szm, rank, sigma=0.3):
    _add_matlab()
    return octave.GauHerPolym3D(szm, rank, sigma)


def matlab_gauss_hermite_moments(img, rank, sigma, normcoef):
    _add_matlab()
    return octave.GauHerMom3Dnorm(img, rank, sigma, normcoef)


def matlab_zernike_polynomials(szm, rank, mask_sphere):
    _add_matlab()
    return octave.zm3dpoly(szm, rank, mask_sphere)


def matlab_zernike_moments(img, rank, mask_sphere, normalization):
    _add_matlab()
    return octave.zm3dmoments(img, rank, mask_sphere, normalization)


def matlab_appell_polynomials_3d(m, n, o, x, y, z, t, s, w):
    '''

    :param m:  rank
    :param n:  rank
    :param o:  rank
    :param x:  coordinates
    :param y:  coordinates
    :param z:  coordinates
    :param t:  type
    :param s:  parameter s of polynomials
    :param w:  weight
    :return:
    '''

    octave.addpath('./matlab')
    return octave.Appell_poly_univ3D(m, n, o,
                                     x, y, z,
                                     t, s, w)


def _add_matlab():
    _path = './matlab'
    assert os.path.exists(_path)
    octave.addpath(_path)


def matlab_cafmi3d(moments2invariants, moments):
    _add_matlab()
    return octave.cafmi3d(moments2invariants, moments, 1)


def matlab_cafmi3dst(coef, ind, moments, types, typeg):
    _add_matlab()
    return octave.cafmi3dst(coef, ind, moments, types, typeg)


def matlab_invariants(img, szm, fname, rank, typec, s, w):
    _add_matlab()
    return octave.Appell_invariant(img, szm, fname, rank, typec, s, w)


def matlab_appell_moments_3D_predef(img, polynomials, m, n, o):
    _path = './matlab'
    assert os.path.exists(_path)
    octave.addpath(_path)
    return octave.Appell_moments_3D_predef(img, polynomials, m, n, o)


def matlab_appell_polynomials_recursive_3d(m, n, o, x, y, z, t, s, w):
    _path = './matlab'
    assert os.path.exists(_path)
    octave.addpath(_path)
    return octave.Appell_poly_univ3Drecursive(m, n, o,
                                              x, y, z,
                                              t, s, w)


def matlab_sphere_harmonic(n, m, theta, phi):
    _add_matlab()
    return octave.spherical_harmonic(n, m, theta, phi)


def matlab_cafmi3dcomplex(f, moments, types, typeg):
    _add_matlab()
    return octave.cafmi3dcomplex(f,
                                 moments,
                                 types,
                                 typeg)

