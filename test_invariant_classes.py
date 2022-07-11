import torch
import numpy as np
from unittest import TestCase
from numpy.testing import assert_allclose
from matlab_bridge import get_images

from invariant3d import AppellInvariant3D, Invariant3D, GauseHermiteInvariants3D
from appell_polynomials_3D import Appell_Type, Appell_polynomial_weights

MAX_RANK = 6
SPHERE_RADIUS = 20
SRZ = 2 * SPHERE_RADIUS + 1
APPELL_PARAMETER_S = 1
NUM_INVARIANTS = 77
TYPES = 0
DEVICE = 'cuda'


def _torch_images():
    return torch.from_numpy(
        np.array(list(get_images()), dtype=np.float64)
        / 255.0
    ).to(torch.device(DEVICE))


def _test_polynomials(model: Invariant3D):
    python_polynomials = model.polynomials.cpu().numpy()
    rank_size = model.max_rank + 1
    srz = model.cube_side
    matlab_polynomials = (
        model._get_matlab_polynomials()
        .reshape((rank_size, rank_size, rank_size,
                  srz, srz, srz), order='F')
        .reshape(rank_size, rank_size, rank_size, srz ** 3, order='C')
    )
    assert_allclose(python_polynomials,
                    matlab_polynomials)


def _test_moments(model: Invariant3D):
    images = _torch_images()
    matlab_moments = model._get_matlab_moments(images)
    assert_allclose(model.get_moments(images).cpu().numpy(),
                   matlab_moments)


def _test_invariants(model: Invariant3D):
    images = _torch_images()
    out = torch.zeros(([images.shape[0], model.num_invariants]), dtype=torch.float64).to(torch.device(DEVICE))
    assert_allclose(model.invariants(images, out=out).cpu().numpy(),
                    model._get_matlab_invariants(images))


def _test_appell(_test_fnc):
    for t in Appell_Type:
        for w in Appell_polynomial_weights:
            model = AppellInvariant3D(appell_type=t,
                                      appell_weight=w,
                                      appell_parameter_s=APPELL_PARAMETER_S,
                                      appell_type_s=TYPES,
                                      num_invariants=NUM_INVARIANTS,
                                      cube_side=SRZ,
                                      max_rank=MAX_RANK,
                                      device=torch.device(DEVICE))
            _test_fnc(model)


def _test_gauss_hermite(_test_fnc):
    model = GauseHermiteInvariants3D(sigma=0.3,
                                     normcoef=0,
                                     normsize=1,
                                     num_invariants=NUM_INVARIANTS,
                                     cube_side=SRZ,
                                     max_rank=MAX_RANK,
                                     device=torch.device(DEVICE))
    _test_fnc(model)


class TestAppellInvariant(TestCase):
    def test_polynomials(self):
        _test_appell(_test_polynomials)

    def test_moments(self):
        _test_appell(_test_moments)

    def test_invariants(self):
        _test_appell(_test_invariants)


class TestGaussHermiteInvariants3D(TestCase):
    def test_moments(self):
        _test_gauss_hermite(_test_moments)
