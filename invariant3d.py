from typing import Tuple, Sequence
import numpy as np
import torch
import enum
from scipy.special import sph_harm, eval_legendre, factorial
from abc import ABC, abstractmethod
from appell_polynomials_3D import appell_polynomials_recursive_3d, Appell_Type, \
    Appell_polynomial_weights
# Appell
from matlab_bridge import matlab_appell_polynomials_recursive_3d, matlab_appell_moments_3D_predef, matlab_cafmi3dst
# Gauss-Hermite
from matlab_bridge import matlab_gauss_hermite_moments, matlab_gauss_hermite_polynoms
from scipy.special import gammaln
# Zernike
from matlab_bridge import matlab_zernike_polynomials, matlab_cafmi3dcomplex, matlab_readinv3dst, matlab_zernike_moments
# Geometric
from matlab_bridge import matlab_geometric_polynomials, matlab_geometric_moments, matlab_complex_polynomials, \
    matlab_complex_moments


# TODO: Doc


class Invariant3D(ABC):
    def __init__(self,
                 typeg: int,
                 num_invariants: int,
                 cube_side: int,
                 max_rank: int,
                 device: torch.device):
        self.typeg = typeg
        # Invariant normalization 0 - none, 1 - to weight, 2 - to degree
        assert self.typeg in [0, 1, 2]
        self.num_invariants = num_invariants
        self.max_rank = max_rank
        self.cube_side = cube_side
        self.polynomials = torch.from_numpy(
            self.init_polynomials()
        ).to(device)
        assert self.polynomials.ndim == 4, "Expecting shape [rank, rank, rank, x*y*z (flatten='C')]"
        assert self.polynomials.shape[0] == self.get_polynomial_shape()[0]
        assert self.polynomials.shape[1] == self.get_polynomial_shape()[1]
        assert self.polynomials.shape[2] == self.get_polynomial_shape()[2]
        assert self.polynomials.shape[3] == self.cube_side ** 3
        self.invariants_ind, self.invariants_coef = self.init_moments2invariants()
        # Move to device
        self.invariants_ind = [x.to(device) for x in self.invariants_ind]
        self.invariants_coef = [x.to(device) for x in self.invariants_coef]
        # Normalization parameters
        self.invariant_weights = self.init_invariant_weights().to(device)
        self.invariant_degrees = self.init_invariant_degrees().to(device)
        # TODO: assert invariants
        assert self.invariants_ind[0].ndim == 3

    def get_moments(self, images: torch.Tensor) -> torch.Tensor:
        assert images.ndim == 4, "Expecting shape [batch_size, x*y*z (flatten='C')]"
        assert images.shape[1] * images.shape[2] * images.shape[3] == self.image_size()
        assert images.device == self.polynomials.device

        return self.normalization_moments(
            torch.sum(
                self.polynomials
                *
                images.reshape((images.shape[0], 1, 1, 1, self.image_size())),
                dim=-1)
        )

    def get_coords(self, lower=-1, upper=1, order='F') -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        coords = np.linspace(lower, upper, self.cube_side, dtype=np.float64)
        [x, y, z] = np.meshgrid(coords, coords, coords)
        x = x.flatten(order=order)
        y = y.flatten(order=order)
        z = z.flatten(order=order)
        return x, y, z

    def image_size(self) -> int:
        return self.cube_side ** 3

    @abstractmethod
    def get_polynomial_shape(self):
        pass

    @abstractmethod
    def init_moments2invariants(self) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        pass

    @abstractmethod
    def init_polynomials(self) -> np.ndarray:
        pass

    @abstractmethod
    def pre_invariant_moments_normalization(self, moments: torch.Tensor) -> torch.Tensor:
        pass

    def normalization_invariants(self, invariants: torch.Tensor) -> torch.Tensor:
        # No normalization
        if self.typeg == 0:
            return invariants
        # Invariant normalization to weight
        elif self.typeg == 1:
            return torch.sign(invariants) * (torch.abs(invariants) ** (1.0 / self.invariant_weights))
        # Invariant normalization to degree
        elif self.typeg == 2:
            return torch.sign(invariants) * (torch.abs(invariants) ** (1.0 / self.invariant_degrees))

    @abstractmethod
    def normalization_moments(self, moments: torch.Tensor) -> torch.Tensor:
        pass

    # Classes for debugging
    @abstractmethod
    def _get_matlab_polynomials(self) -> np.ndarray:
        """For testing purposes"""
        pass

    @abstractmethod
    def _get_matlab_moments(self, images: torch.Tensor) -> np.ndarray:
        """For testing purposes"""
        pass

    @abstractmethod
    def _get_matlab_invariants(self, images: torch.Tensor) -> np.ndarray:
        """For testing purposes"""
        pass

    def invariants(self, images: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
        moments = self.get_moments(images=images)
        moments = self.pre_invariant_moments_normalization(moments=moments)
        assert out.shape[0] == moments.shape[0]
        generator = enumerate(zip(self.invariants_ind, self.invariants_coef))
        for idx, (inv_idx, inv_coef) in generator:
            products = moments[:,
                       inv_idx[..., 0],
                       inv_idx[..., 1],
                       inv_idx[..., 2]
                       ]
            out[:, idx] = torch.real(
                torch.sum(torch.prod(products, dim=-2) * inv_coef, dim=-1)
            )

        # TODO: post_invariant normalization
        return self.normalization_invariants(out)

    @abstractmethod
    def init_invariant_weights(self):
        pass

    @abstractmethod
    def init_invariant_degrees(self):
        pass


class CafmidstInvariant3D(Invariant3D):

    def __init__(self, typeg: int, types: int, num_invariants: int, cube_side: int, max_rank: int,
                 device: torch.device):
        self._polynomial_shape = (max_rank + 1, max_rank + 1, max_rank + 1)
        self.types = types
        self.param_v = torch.tensor(2 + (types % 2) if types > 0 else 3,
                                    dtype=torch.float64,
                                    device=device)
        super().__init__(typeg, num_invariants, cube_side, max_rank, device)
        self.index_sum = torch.from_numpy(
            np.sum(np.indices(self.get_polynomial_shape()), axis=0).astype(np.float64)
        ).to(device)
        self.v = torch.tensor(data=(self.types % 2) + 2, dtype=torch.float64)

    @abstractmethod
    def init_polynomials(self) -> np.ndarray:
        pass

    @abstractmethod
    def normalization_moments(self, moments: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_matlab_polynomials(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_matlab_moments(self, images: torch.Tensor) -> np.ndarray:
        pass

    def get_polynomial_shape(self):
        return self._polynomial_shape

    def init_moments2invariants(self) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        moments2invariants = np.load("torch_invariants3Dinv10/moments2invariants.npz",
                                     allow_pickle=True)
        invariant_ind = list(moments2invariants['ind'][0][:self.num_invariants])
        invariant_coef = list(moments2invariants['coef'][0][0][:self.num_invariants])

        # Test uint8
        for ind in invariant_ind:
            assert np.max(ind) <= 255

        # Convert to Tensor
        invariant_coef = [torch.from_numpy(np.array(coef, dtype=np.float64)) for coef in invariant_coef]
        invariant_ind = [torch.from_numpy(ind.astype(int)) for ind in invariant_ind]
        return invariant_ind, invariant_coef

    def pre_invariant_moments_normalization(self, moments: torch.Tensor) -> torch.Tensor:
        if self.types > 0:
            # Volume normalization constant
            m000 = moments[..., 0:1, 0:1, 0:1]
            # Normalization to scaling
            moments /= m000 ** ((self.index_sum + self.v) / self.v)
            if self.types == 1:
                moments *= torch.pi ** (self.index_sum / 6.0) * (self.index_sum + 3.0) / 1.5 ** (
                        self.index_sum / 3.0 + 1.0)
            elif self.types == 2:
                moments *= torch.pi ** (self.index_sum / 4.0) * 2.0 ** (self.index_sum / 2.0)
        return moments

    def _get_matlab_invariants(self, images: torch.Tensor) -> np.ndarray:
        matlab_moments = self._get_matlab_moments(images)
        matlab_invariants = np.zeros((images.shape[0], self.num_invariants))
        # Matlab parameters
        moments2invariants = np.load("rot3Dinv10indep.npz", allow_pickle=True)
        matlab_invariant_ind = list(moments2invariants['ind'][0][:self.num_invariants])
        matlab_invariant_coef = list(moments2invariants['coef'][0][:self.num_invariants])
        for idx, img_moments in enumerate(matlab_moments):
            matlab_invariants[idx] = matlab_cafmi3dst(matlab_invariant_coef,
                                                      matlab_invariant_ind,
                                                      img_moments,
                                                      types=self.types,
                                                      typeg=self.typeg)
        return matlab_invariants

    def init_invariant_degrees(self):
        return torch.tensor(data=[ind.shape[0] for ind in self.invariants_ind],
                            dtype=torch.float64)

    def init_invariant_weights(self):
        return torch.tensor(data=[torch.sum(ind[:, 0, :]) / self.param_v for ind in self.invariants_ind],
                            dtype=torch.float64)


class CafmidComplexInvariant3D(Invariant3D):
    def __init__(self,
                 types: int,
                 typeg: int,
                 num_invariants: int,
                 cube_side: int,
                 max_rank: int,
                 device: torch.device):
        self.types = types
        self._polynomial_shape = (max_rank + 1,
                                  np.floor(max_rank / 2.0).astype(int) + 1,
                                  2 * max_rank + 1)
        super().__init__(typeg, num_invariants, cube_side, max_rank, device)

    @abstractmethod
    def init_polynomials(self) -> np.ndarray:
        pass

    @abstractmethod
    def normalization_moments(self, moments: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def _get_matlab_polynomials(self) -> np.ndarray:
        pass

    @abstractmethod
    def _get_matlab_moments(self, images: torch.Tensor) -> np.ndarray:
        pass

    def get_polynomial_shape(self):
        return self._polynomial_shape

    def init_moments2invariants(self) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        invariant_ind = [ind.astype(np.int64) for ind in
                         np.load('complex_moments2invariants/complex_moments2invariants_ind.npz', allow_pickle=True)[
                             'ind']]
        invariant_coef = list(
            np.load('complex_moments2invariants/complex_moments2invariants_coef.npz', allow_pickle=True).values())

        # Test uint8
        for ind in invariant_ind:
            assert np.max(ind) <= 255

        # Convert to Tensor
        invariant_coef = [torch.from_numpy(np.array(coef, dtype=np.float64)) for coef in invariant_coef]
        invariant_ind = [torch.from_numpy(ind.astype(int)) for ind in invariant_ind]
        return invariant_ind, invariant_coef

    def pre_invariant_moments_normalization(self, moments: torch.Tensor) -> torch.Tensor:
        if self.types > 0:
            # NOTE: normalization is inplace
            raise NotImplementedError("Not implemented")
        else:
            return moments

    def init_invariant_weights(self):
        return torch.tensor(data=[torch.sum(ind[:, 0, 0], dtype=torch.float64) / 3.0 for ind in self.invariants_ind],
                            dtype=torch.float64)

    def init_invariant_degrees(self):
        return torch.tensor(data=[ind.shape[0] for ind in self.invariants_ind],
                            dtype=torch.float64)

    def _get_matlab_invariants(self, images: torch.Tensor) -> np.ndarray:
        matlab_moments = self._get_matlab_moments(images)
        matlab_invariants = np.zeros((images.shape[0], self.num_invariants))
        # Matlab parameters
        indices = matlab_readinv3dst("matlab/ccmf6indep.txt")
        for idx, img_moments in enumerate(matlab_moments):
            matlab_invariants[idx] = matlab_cafmi3dcomplex(indices,
                                                           img_moments,
                                                           types=self.types,
                                                           typeg=self.typeg)
        return matlab_invariants


class AppellInvariant3D(CafmidstInvariant3D):

    def __init__(self,
                 appell_type: Appell_Type,
                 appell_weight: Appell_polynomial_weights,
                 appell_parameter_s: float,
                 appell_type_s: int,
                 typeg: int,
                 num_invariants: int,
                 cube_side: int,
                 max_rank: int,
                 device: torch.device):
        self.appell_type = appell_type
        self.appell_weight = appell_weight
        # TODO: Make Tensor
        self.appell_parameter_s = appell_parameter_s
        self.appell_type_s = appell_type_s
        self.normalization_factor = torch.tensor(data=((cube_side - 1) / 2) ** 3,
                                                 dtype=torch.float64,
                                                 device=device)
        # Convert to Tensor
        super().__init__(typeg=typeg,
                         types=appell_type_s,
                         num_invariants=num_invariants,
                         cube_side=cube_side,
                         max_rank=max_rank,
                         device=device)

    def init_polynomials(self):
        x, y, z = self.get_coords(lower=-1, upper=1, order='F')
        rank_size = self.max_rank + 1
        srz = self.cube_side
        return (
            appell_polynomials_recursive_3d(m=self.max_rank,
                                            n=self.max_rank,
                                            o=self.max_rank,
                                            x=x, y=y, z=z,
                                            appell_type=self.appell_type,
                                            weight=self.appell_weight,
                                            s=self.appell_parameter_s)
            # Matlab format
            .reshape((rank_size, rank_size, rank_size,
                      srz, srz, srz), order='F')
            # Pytorch format
            .reshape((rank_size, rank_size, rank_size,
                      srz ** 3))
        )

    def _get_matlab_polynomials(self) -> np.ndarray:
        x, y, z = self.get_coords(lower=-1, upper=1, order='F')
        return (
            matlab_appell_polynomials_recursive_3d(self.max_rank, self.max_rank, self.max_rank,
                                                   x, y, z,
                                                   self.appell_type.value,
                                                   self.appell_parameter_s,
                                                   self.appell_weight.value)
            .reshape((self.max_rank + 1, self.max_rank + 1, self.max_rank + 1,
                      self.cube_side, self.cube_side, self.cube_side), order='F')
        )

    def _get_matlab_moments(self, images: torch.Tensor):
        matlab_polynomials = self._get_matlab_polynomials()
        matlab_moments = np.zeros([images.shape[0], self.max_rank + 1, self.max_rank + 1, self.max_rank + 1])
        for idx, image in enumerate(images):
            matlab_moments[idx] = (
                matlab_appell_moments_3D_predef(img=image.cpu().numpy(),
                                                polynomials=matlab_polynomials,
                                                m=self.max_rank,
                                                n=self.max_rank,
                                                o=self.max_rank)
            )
        return matlab_moments

    def normalization_moments(self, moments: torch.Tensor):
        return moments / self.normalization_factor


class GaussHermiteInvariants3D(CafmidstInvariant3D):
    def __init__(self,
                 typeg: int,
                 types: int,
                 sigma: float,
                 normcoef: float,
                 num_invariants: int,
                 cube_side: int,
                 max_rank: int,
                 device: torch.device):
        self.sigma = sigma
        self.normcoef = normcoef
        # TODO: Add parameters
        super().__init__(typeg=typeg,
                         types=types,
                         num_invariants=num_invariants,
                         cube_side=cube_side,
                         max_rank=max_rank,
                         device=device)
        _weights = np.zeros((self.max_rank + 1, self.max_rank + 1, self.max_rank + 1), dtype=np.float64)
        for rx in range(self.max_rank + 1):
            for ry in range(self.max_rank + 1):
                for rz in range(self.max_rank + 1):
                    _weights[rx, ry, rz] = np.exp(gammaln(rx + 1) + gammaln(ry + 1) + gammaln(rz + 1) - gammaln(
                        rx + ry + rz + 1) * normcoef / 2 - gammaln((rx + ry + rz) / 2 + 1) * (1 - normcoef))
        # Normalization
        _weights = _weights * (2.0 / self.cube_side) ** 3
        self.moments_weights = torch.from_numpy(_weights[np.newaxis, ...]).to(device)

    def init_polynomials(self) -> np.ndarray:
        # TODO: refactors
        x, y, z = self.get_coords(lower=-1, upper=1, order='F')
        # Norm by sigma
        x = x / self.sigma
        y = y / self.sigma
        z = z / self.sigma
        op_type = np.float64

        def _ker_norm(ker, v):
            ker[0, :] = np.exp(-np.power(v, 2, dtype=op_type) / 2.0, dtype=op_type) / np.power(np.pi, 0.25,
                                                                                               dtype=op_type)
            ker[1, :] = np.power(2, 0.5, dtype=op_type) * v * ker[0, :]

            for d in range(2, self.max_rank + 1):
                a = (1.0 / np.sqrt(2.0 * d, dtype=op_type) * v * ker[d - 1, :])
                b = (- np.sqrt(1.0 - 1.0 / d, dtype=op_type) * ker[d - 2, :])
                ker[d, :] = a + b
            return ker

        kerx = _ker_norm(ker=np.zeros((self.max_rank + 1, len(x)), dtype=op_type),
                         v=x)
        kery = _ker_norm(ker=np.zeros((self.max_rank + 1, len(y)), dtype=op_type),
                         v=y)
        kerz = _ker_norm(ker=np.zeros((self.max_rank + 1, len(z)), dtype=op_type),
                         v=z)

        polynomials = np.zeros((self.max_rank + 1, self.max_rank + 1, self.max_rank + 1,
                                self.cube_side ** 3))

        for rx in range(0, self.max_rank + 1):
            for ry in range(0, self.max_rank + 1 - rx):
                for rz in range(0, self.max_rank + 1 - rx - ry):
                    for a in range(0, self.cube_side ** 3):
                        polynomials[rx, ry, rz, a] = (
                                kerx[rx, a] * kery[ry, a] * kerz[rz, a]
                        )
        return (
            polynomials
            .reshape((self.max_rank + 1, self.max_rank + 1, self.max_rank + 1,
                      self.cube_side, self.cube_side, self.cube_side), order='F')
            .reshape((self.max_rank + 1, self.max_rank + 1, self.max_rank + 1, self.cube_side ** 3), order='C')
        )

    def normalization_moments(self, moments: torch.Tensor) -> torch.Tensor:
        return moments * self.moments_weights

    def _get_matlab_polynomials(self) -> np.ndarray:
        return matlab_gauss_hermite_polynoms(szm=self.cube_side, rank=self.max_rank, sigma=self.sigma)

    def _get_matlab_moments(self, images: torch.Tensor) -> np.ndarray:
        matlab_moments = np.zeros([images.shape[0], self.max_rank + 1, self.max_rank + 1, self.max_rank + 1])
        for idx, image in enumerate(images):
            matlab_moments[idx] = (
                matlab_gauss_hermite_moments(img=image.cpu().numpy(),
                                             rank=self.max_rank,
                                             sigma=self.sigma,
                                             normcoef=self.normcoef))

        return matlab_moments


class ZernikeMomentsNormalization(enum.Enum):
    WEIGHT = 0  # SZM ^ 3
    M00 = 1


class ZernikeInvariants3D(CafmidComplexInvariant3D):
    def __init__(self,
                 typeg: int,
                 types: int,
                 num_invariants: int,
                 cube_side: int,
                 max_rank: int,
                 mask_sphere: bool,
                 moment_normalization: ZernikeMomentsNormalization,
                 device: torch.device):
        self.types = types
        self.moment_normalization = moment_normalization
        self.mask_sphere = mask_sphere

        super().__init__(typeg=typeg,
                         types=types,
                         num_invariants=num_invariants,
                         cube_side=cube_side,
                         max_rank=max_rank,
                         device=device)
        self.moment_normalization_parameter = torch.tensor(data=self.cube_side ** 3, device=device)

    def init_polynomials(self) -> np.ndarray:
        x, y, z = self.get_coords(-1, 1, order='F')

        # Spherical coordinate system
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2, dtype=np.float64)
        theta = np.arccos(np.divide(z, r, where=r != 0, dtype=np.float64), dtype=np.float64)
        # Replace the r=0 NaN value
        assert np.sum(r == 0) == 1
        theta[r == 0] = 0.0
        phi = np.arctan2(y, x, dtype=np.float64)
        out_of_sphere = (r > 1.0)

        # Kintner (recursive) method
        # https://en.wikipedia.org/wiki/Zernike_polynomials#Zernike_polynomials
        polynomials = np.zeros((*self.get_polynomial_shape(), self.cube_side ** 3),
                               dtype=np.complex128)

        for el in range(0, self.max_rank + 1):
            for em in range(-el, el + 1):
                rmn0 = r ** el
                # NOTE: scipy.special.sph_harm(m, n, phi, theta) matlab.spherical_harmonics(n, m, theta, phi)
                vmn = np.conjugate(rmn0 * self.sphere_harmonics(m=em, n=el, phi=theta, theta=phi))
                polynomials[el, np.floor(el / 2).astype(int), em + el, :] = (el + 1) / np.pi * vmn
                if (self.max_rank - el) >= 2:
                    rmn2 = (el + 2) * r ** (el + 2) - (el + 1) * r ** el
                    vmn = np.conjugate(rmn2 * self.sphere_harmonics(m=em, n=el, phi=theta, theta=phi))
                    polynomials[el + 2, np.floor(el / 2).astype(int), em + el, :] = (el + 3) / np.pi * vmn
                else:
                    # TODO: wtf rmn2
                    pass
                for es in range(el + 4, self.max_rank + 1, 2):
                    k1 = (es + el) * (es - el) * (es - 2) / 2
                    k2 = 2 * es * (es - 1) * (es - 2)
                    k3 = -el ** 2 * (es - 1) - es * (es - 1) * (es - 2)
                    k4 = -es * (es + el - 2) * (es - el - 2) / 2
                    rmn4 = ((k2 * r ** 2 + k3) * rmn2 + k4 * rmn0) / k1
                    vmn = np.conjugate(rmn4 * self.sphere_harmonics(m=em, n=el, phi=theta, theta=phi))
                    polynomials[es, np.floor(el / 2).astype(int), em + el, :] = (es + 1) / np.pi * vmn
                    rmn0 = rmn2
                    rmn2 = rmn4

        # Mask values of the sphere
        if self.mask_sphere:
            polynomials[:, :, :, out_of_sphere] = 0.0

        return (polynomials
                # Matlab format
                .reshape((*self.get_polynomial_shape(),
                          self.cube_side, self.cube_side, self.cube_side), order='F')
                # Pytorch format
                .reshape((*self.get_polynomial_shape(),
                          self.cube_side ** 3))
                )

    def normalization_moments(self, moments: torch.Tensor) -> torch.Tensor:
        if self.moment_normalization == ZernikeMomentsNormalization.M00:
            return moments / moments[:, 0:1, 0:1, 0:1]
        elif self.moment_normalization == ZernikeMomentsNormalization.WEIGHT:
            return moments / self.moment_normalization_parameter

    def _get_matlab_polynomials(self) -> np.ndarray:
        return matlab_zernike_polynomials(self.cube_side, self.max_rank, self.mask_sphere)

    def _get_matlab_moments(self, images: torch.Tensor) -> np.ndarray:
        matlab_moments = np.zeros([images.shape[0], *self.get_polynomial_shape()], dtype=np.complex128)
        for idx, image in enumerate(images):
            matlab_moments[idx] = (
                matlab_zernike_moments(image.cpu().numpy(),
                                       self.max_rank,
                                       self.mask_sphere,
                                       self.moment_normalization.value))

        return matlab_moments

    @staticmethod
    def sphere_harmonics(m, n, theta, phi):
      return np.nan_to_num(sph_harm(m, n, theta, phi, dtype=np.complex128))

    @staticmethod
    def sphere_harmonics_imp(n, m, theta, phi):
        p = eval_legendre(n, np.cos(theta))
        am = np.abs(m, dtype=np.float64)
        y = p[int(np.abs(m)), :]
        y = y * np.cos((am * phi) + 1j * np.sin(np.abs(am) * phi))
        y = y * np.sqrt((2. * n + 1.)/4./np.pi * factorial(n - am) / factorial(n + am))
        if m < 0:
            y = np.conjugate(y) * (-1) ** (-m)
        return y


class GeometricInvariants3D(CafmidstInvariant3D):
    def init_polynomials(self) -> np.ndarray:
        middle = int((self.cube_side + 1) / 2)
        [y, x, z] = self.get_coords(-middle + 1, middle - 1, order='F')

        polynomials = np.zeros([self.max_rank + 1, self.max_rank + 1, self.max_rank + 1,
                                self.cube_side ** 3])

        for p in range(0, self.max_rank + 1):
            for q in range(0, self.max_rank - p + 1):
                for r in range(0, self.max_rank - p - q + 1):
                    polynomials[p, q, r, :] = x ** p * y ** q * z ** r

        return (polynomials
                # Matlab format
                .reshape((*self.get_polynomial_shape(),
                          self.cube_side, self.cube_side, self.cube_side), order='F')
                # Pytorch format
                .reshape((*self.get_polynomial_shape(),
                          self.cube_side ** 3))
                )

    def normalization_moments(self, moments: torch.Tensor) -> torch.Tensor:
        return moments

    def _get_matlab_polynomials(self) -> np.ndarray:
        # Typec=1 center of cube
        return matlab_geometric_polynomials(szm=self.cube_side, order=self.max_rank, typec=1)

    def _get_matlab_moments(self, images: torch.Tensor) -> np.ndarray:
        matlab_moments = np.zeros([images.shape[0], self.max_rank + 1, self.max_rank + 1, self.max_rank + 1])
        for idx, image in enumerate(images):
            matlab_moments[idx] = (
                # Typec=1 center of cube
                matlab_geometric_moments(image.cpu().numpy(), self.max_rank, typec=1)
            )

        return matlab_moments


class ComplexInvariants3D(CafmidComplexInvariant3D):
    def init_polynomials(self) -> np.ndarray:
        x, y, z = self.get_coords(-1, 1, order='F')

        # Spherical coordinate system
        r = np.sqrt(x ** 2 + y ** 2 + z ** 2, dtype=np.float64)
        theta = np.arccos(np.divide(z, r,
                                    where=r != 0,
                                    dtype=np.float64),
                          dtype=np.float64)
        # Replace the r=0 NaN value
        assert np.sum(r == 0) == 1
        theta[r == 0] = 0.0
        phi = np.arctan2(y, x, dtype=np.float64)

        polynomials = np.zeros((*self.get_polynomial_shape(),
                                self.cube_side ** 3),
                               dtype=np.complex128)

        for es in range(0, self.max_rank + 1):
            for el in range(es % 2, es + 1, 2):
                for em in range(-el, el + 1):
                    # NOTE: scipy.special.sph_harm(m, n, phi, theta) matlab.spherical_harmonics(n, m, theta, phi)
                     polynomials[es,
                                 np.floor(el / 2.0).astype(int),
                                 em + el, :] = np.power(r, es, dtype=np.complex128) * ZernikeInvariants3D.sphere_harmonics(m=em,
                                                                                                                           n=el,
                                                                                                                           phi=theta,
                                                                                                                           theta=phi)


        return (polynomials
                # Matlab format
                .reshape((*self.get_polynomial_shape(),
                          self.cube_side, self.cube_side, self.cube_side), order='F')
                # Pytorch format
                .reshape((*self.get_polynomial_shape(),
                          self.cube_side ** 3))
                )

    def normalization_moments(self, moments: torch.Tensor) -> torch.Tensor:
        return moments

    def _get_matlab_polynomials(self) -> np.ndarray:
        return matlab_complex_polynomials(szm=self.cube_side,
                                          order=self.max_rank)

    def _get_matlab_moments(self, images: torch.Tensor) -> np.ndarray:
        matlab_moments = np.zeros([images.shape[0], *self.get_polynomial_shape()], dtype=np.complex128)
        for idx, image in enumerate(images):
            matlab_moments[idx] = matlab_complex_moments(image.cpu().numpy(),
                                                         self.max_rank,
                                                         norm=3)
        return matlab_moments
