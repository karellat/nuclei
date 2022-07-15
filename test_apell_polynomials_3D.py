import time
from matlab_bridge import *
from scipy.special import sph_harm
import torch
from glob import glob
import numpy as np
from unittest import TestCase
from appell_polynomials_3D import appell_polynomials_recursive_3d, appell_polynomials_3d, pochham, Appell_Type, \
    Appell_polynomial_weights, appell_moments_3d_predef, cafmi3d, moments_volume_normalization, \
    get_all_appell_polynomials_3d, cafmi3d_torch
from appell_invariant import InvariantAppell
from invariant3d import ZernikeInvariants3D

MATLAB_ROOT_DIR = '.'
MAX_RANK = 6
SPHERE_RADIUS = 20
APPELL_PARAMETER_S = 1


class Test(TestCase):
    # TODO: Flatten
    def test_appell_polynomials_consistency(self):
        for t in Appell_Type:
            for w in Appell_polynomial_weights:
                srz = 2 * SPHERE_RADIUS + 1
                coords = np.linspace(-1, 1, srz)
                [x, y, z] = np.meshgrid(coords, coords, coords)
                vl = np.min([x.size, y.size, z.size])
                polynomial = np.zeros((MAX_RANK + 1, MAX_RANK + 1, MAX_RANK + 1, vl))
                x = x.flatten(order='F')[:, np.newaxis]
                y = y.flatten(order='F')[:, np.newaxis]
                z = z.flatten(order='F')[:, np.newaxis]
                for ord in range(MAX_RANK + 1):
                    for p in range(ord + 1):
                        for q in range(ord - p + 1):
                            r = ord - p - q
                            polynomial[p, q, r] = matlab_appell_polynomials_3d(p, q, r,
                                                                               x, y, z,
                                                                               t.value, APPELL_PARAMETER_S,
                                                                               w.value).flatten(order='F')

                recursive = matlab_appell_polynomials_recursive_3d(MAX_RANK, MAX_RANK, MAX_RANK,
                                                                   x, y, z,
                                                                   t.value, APPELL_PARAMETER_S, w.value)
                np.testing.assert_allclose(polynomial, recursive,
                                           err_msg=f"Type {t}, weights {w}", )

    def test_appell_polynomials_3d(self):
        for t in Appell_Type:
            for w in Appell_polynomial_weights:
                srz = 2 * SPHERE_RADIUS + 1
                coords = np.linspace(-1, 1, srz)
                [x, y, z] = np.meshgrid(coords, coords, coords)
                matlab_t = 1 if t == Appell_Type.U else 0
                if w == Appell_polynomial_weights.NO_WEIGHT:
                    matlab_w = 0
                elif w == Appell_polynomial_weights.WEIGHTED:
                    matlab_w = 1
                elif w == Appell_polynomial_weights.RANK_NORM:
                    matlab_w = 2
                elif w == Appell_polynomial_weights.WEIGHTED_3:
                    matlab_w = 3
                elif w == Appell_polynomial_weights.WEIGHTED_4:
                    matlab_w = 4
                elif w == Appell_polynomial_weights.WEIGHTED_5:
                    matlab_w = 5
                else:
                    raise NotImplementedError()
                matlab_poly = matlab_appell_polynomials_3d(MAX_RANK, MAX_RANK, MAX_RANK,
                                                           x, y, z,
                                                           matlab_t, APPELL_PARAMETER_S, matlab_w)

                python_poly = appell_polynomials_3d(MAX_RANK, MAX_RANK, MAX_RANK,
                                                    x, y, z,
                                                    t, APPELL_PARAMETER_S, w)
                np.testing.assert_allclose(matlab_poly, python_poly,
                                           err_msg=f"Type {t}, weights {w}")

    def test_appell_polynomials_recursive(self):
        for t in Appell_Type:
            for w in Appell_polynomial_weights:
                srz = 2 * SPHERE_RADIUS + 1
                coords = np.linspace(-1, 1, srz)
                [x, y, z] = np.meshgrid(coords, coords, coords)
                x = x.reshape((x.size, 1))
                y = y.reshape((y.size, 1))
                z = z.reshape((z.size, 1))
                matlab_poly = matlab_appell_polynomials_recursive_3d(MAX_RANK, MAX_RANK, MAX_RANK,
                                                                     x, y, z, t.value, APPELL_PARAMETER_S, w.value)

                python_poly = appell_polynomials_recursive_3d(MAX_RANK, MAX_RANK, MAX_RANK,
                                                              np.squeeze(x),
                                                              np.squeeze(y),
                                                              np.squeeze(z),
                                                              t, APPELL_PARAMETER_S, w)
                np.testing.assert_allclose(matlab_poly, python_poly,
                                           err_msg=f"Type {t}, weights {w}")

    def test_pochham(self):
        for _ in range(1000):
            x = np.random.randint(0, 10, np.random.randint(0, 100)).astype(float)
            k = float(np.random.randint(0, 100))
            matlab_res = np.squeeze(matlab_pochham(x, k))
            python_res = np.squeeze(pochham(x, k))
            np.testing.assert_allclose(matlab_res, python_res)
        np.testing.assert_allclose(
            np.squeeze(pochham(-1, 2 * 1)),
            np.squeeze(matlab_pochham(-1, 2 * 1)))

    #    def test_load_rot3d(self):
    # TODO: Fix no end reading
    #        _path = '/home/karellat/PycharmProjects/nuclei/matlab/rot3Dinv10indep.txt'
    #        f = matlab_readinv3dst(_path)
    #        np.savez_compressed("rot3Dinv10indep.npz", coef=f.coef, ind=f.ind)

    def test_appell_moments(self):
        appell_rank = 6
        appell_weight = Appell_polynomial_weights.WEIGHTED_4
        appell_type = Appell_Type.U
        appell_param = +1
        img = next(iter(get_images()))
        srz = img.shape[0]
        coords = np.linspace(-1, 1, srz)
        [x, y, z] = np.meshgrid(coords, coords, coords)
        x = x.flatten(order='F')
        y = y.flatten(order='F')
        z = z.flatten(order='F')
        matlab_polynomials = matlab_appell_polynomials_recursive_3d(appell_rank, appell_rank, appell_rank,
                                                                    x[..., np.newaxis],
                                                                    y[..., np.newaxis],
                                                                    z[..., np.newaxis],
                                                                    t=appell_type.value,
                                                                    s=appell_param,
                                                                    w=appell_weight.value)
        python_polynomials = appell_polynomials_recursive_3d(appell_rank, appell_rank, appell_rank,
                                                             x, y, z,
                                                             appell_type=appell_type,
                                                             s=appell_param,
                                                             weight=appell_weight)
        np.testing.assert_allclose(matlab_polynomials, python_polynomials)
        imgs = torch.from_numpy(np.array(list(get_images()), dtype=np.float64))
        # Python moments
        torch_polynomials = torch.from_numpy(
            python_polynomials.reshape((7, 7, 7, 41, 41, 41), order='F').reshape(7, 7, 7, 41 ** 3))
        python_moments = appell_moments_3d_predef(imgs, torch_polynomials)
        for idx, img in enumerate(get_images()):
            matlab_moments = matlab_appell_moments_3D_predef(img.astype(np.float64), matlab_polynomials,
                                                             appell_rank, appell_rank, appell_rank)
            np.testing.assert_allclose(matlab_moments, python_moments[idx].numpy())

    def test_invariants(self):
        appell_rank = 6
        appell_weight = Appell_polynomial_weights.WEIGHTED_4
        appell_type = Appell_Type.U
        appell_param = +1
        types = 0
        img = next(iter(get_images()))
        srz = img.shape[0]
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

        python_polynomials = appell_polynomials_recursive_3d(appell_rank, appell_rank, appell_rank,
                                                             x, y, z,
                                                             appell_type=appell_type,
                                                             s=appell_param,
                                                             weight=appell_weight)

        np.testing.assert_allclose(matlab_polynomials, python_polynomials)
        # TODO: Forloop over images
        i = 1
        # Moments
        matlab_moments = np.zeros((4, 7, 7, 7))
        imgs = np.array(list(get_images()))
        for idx, img in enumerate(get_images()):
            matlab_moments[idx] = matlab_appell_moments_3D_predef(img.astype(np.float64), matlab_polynomials,
                                                                  appell_rank, appell_rank, appell_rank)

        torch_imgs = torch.from_numpy(imgs.astype(np.float64))
        # Change ordering of polynomials
        begin = time.time()
        torch_polynomials = torch.from_numpy(
            python_polynomials.reshape((7, 7, 7, 41, 41, 41), order='F').reshape(7, 7, 7, 41 ** 3))
        print(f"Elapsed time {time.time() - begin}")
        python_moments = appell_moments_3d_predef(torch_imgs, torch_polynomials, np.min(imgs[0].shape))
        np.testing.assert_allclose(matlab_moments, python_moments)
        matlab_invariants = np.zeros((4, 77))
        python_invariants = np.zeros((4, 77))
        # Invariants
        if types > 0:
            v = 2 + (types % 2)
        else:
            v = 3

        # Prepare invariants parameters
        moments2invariants = np.load("rot3Dinv10indep.npz", allow_pickle=True)
        invariant_ind = list(moments2invariants['ind'][0][:77])
        invariant_coef = list(moments2invariants['coef'][0][:77])
        invariant_sizes = np.array([np.sum(ind[0, :]) / v for ind in invariant_ind])

        # Convert to Tensor
        invariant_ind = [torch.from_numpy(ind.astype(int)) for ind in invariant_ind]
        invariant_coef = [torch.from_numpy(coef.astype(np.float64)) for coef in invariant_coef]
        invariant_sizes = torch.from_numpy(invariant_sizes)

        if types > 0:
            python_moments = moments_volume_normalization(python_moments, types)

        for idx in range(python_moments.shape[0]):
            moments = python_moments[idx].clone()
            matlab_invariants[idx] = matlab_cafmi3dst([coef.cpu().numpy() for coef in invariant_coef],
                                                      [inv_idx.cpu().numpy() for inv_idx in invariant_ind],
                                                      moments.cpu().numpy(),
                                                      types=types,
                                                      typeg=1)
        # NOTE: moments_volume_normalization changes the moments variable

        for idx in range(python_moments.shape[0]):
            python_invariants[idx] = cafmi3d(python_moments[idx],
                                             invariant_ind=invariant_ind,
                                             invariant_coef=invariant_coef,
                                             invariant_sizes=invariant_sizes)

        for idx, img in enumerate(get_images()):
            inv = matlab_image_to_invariants(img,
                                             srz=srz,
                                             types=types,
                                             appell_rank=appell_rank,
                                             appell_type=appell_type,
                                             appell_param=appell_param,
                                             appell_weight=appell_weight)
            np.testing.assert_allclose(np.squeeze(inv), matlab_invariants[idx])

        np.testing.assert_allclose(python_invariants,
                                   np.squeeze(matlab_invariants))
        #  rtol=0.1e-05, atol=0.1e-07)

    def test_pytorch_invariants(self):
        appell_rank = 6
        appell_weight = Appell_polynomial_weights.WEIGHTED_4
        appell_type = Appell_Type.U
        appell_param = +1
        invariants_num = 77
        types = 0
        img = next(iter(get_images()))
        srz = img.shape[0]
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

        python_polynomials = appell_polynomials_recursive_3d(appell_rank, appell_rank, appell_rank,
                                                             x, y, z,
                                                             appell_type=appell_type,
                                                             s=appell_param,
                                                             weight=appell_weight)

        np.testing.assert_allclose(matlab_polynomials, python_polynomials)
        # TODO: Forloop over images
        i = 1
        # Moments
        matlab_moments = np.zeros((4, 7, 7, 7))
        imgs = np.array(list(get_images()))
        for idx, img in enumerate(get_images()):
            matlab_moments[idx] = matlab_appell_moments_3D_predef(img.astype(np.float64), matlab_polynomials,
                                                                  appell_rank, appell_rank, appell_rank)

        matlab_invariants = np.zeros((4, 77))

        # Matlab paramaters
        moments2invariants = np.load("rot3Dinv10indep.npz", allow_pickle=True)
        matlab_invariant_ind = list(moments2invariants['ind'][0][:77])
        matlab_invariant_coef = list(moments2invariants['coef'][0][:77])

        for idx in range(matlab_moments.shape[0]):
            moments = matlab_moments[idx]
            matlab_invariants[idx] = matlab_cafmi3dst(matlab_invariant_coef,
                                                      matlab_invariant_ind,
                                                      moments,
                                                      types=types,
                                                      typeg=1)
        device = torch.device('cuda')
        torch_imgs = torch.from_numpy(imgs.astype(np.float64)).to(device)
        invariant_calc = InvariantAppell(rank=appell_rank,
                                         appell_type=appell_type,
                                         appell_param=appell_param,
                                         appell_weight=appell_weight,
                                         img_srz=srz,
                                         invariants_num=invariants_num,
                                         types=types,
                                         device=device)
        python_invariants = torch.zeros((4, invariants_num)).to(device)
        python_invariants = invariant_calc.calc_invariants(imgs=torch_imgs,
                                                           out=python_invariants)

        np.testing.assert_allclose(python_invariants.cpu().numpy(),
                                   np.squeeze(matlab_invariants))

    def test_sphere_harmonics(self):
        for n in range(0, 20):
            for m in range(-n, n):
                for theta in np.linspace(0, 2 * np.pi, 10):
                    for phi in np.linspace(0, np.pi, 10):
                        matlab_harmonics = matlab_sphere_harmonic(n=n, m=m, theta=theta, phi=phi)
                        scipy_harmonics = ZernikeInvariants3D.sphere_harmonics(m=m, n=n, phi=theta, theta=phi)
                        np.testing.assert_allclose(matlab_harmonics, scipy_harmonics,
                                                   err_msg=f"n = {n}; m = {m}; theta = {theta}, phi = {phi}",
                                                   rtol=1e-15,
                                                   atol=1e-15)


