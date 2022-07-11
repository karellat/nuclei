from typing import Tuple, Sequence
import numpy as np
import torch
from abc import ABC, abstractmethod
from appell_polynomials_3D import appell_polynomials_recursive_3d, Appell_Type, \
    Appell_polynomial_weights, moments_volume_normalization
# Appell
from matlab_bridge import matlab_appell_polynomials_recursive_3d, matlab_appell_moments_3D_predef, matlab_cafmi3dst
# Gauss-Hermite
from matlab_bridge import matlab_gauss_hermite_moments

#TODO: Doc


class Invariant3D(ABC):
    def __init__(self,
                 num_invariants: int,
                 cube_side: int,
                 max_rank: int,
                 device: torch.device):
        self.num_invariants = num_invariants
        self.max_rank = max_rank
        self.cube_side = cube_side
        # TODO: check types
        self.polynomials = torch.from_numpy(
            self.init_polynomials()
        ).to(device)
        self.invariants_ind, self.invariants_coef = self.init_moments2invariants()
        # Move to device
        self.invariants_ind = [x.to(device) for x in self.invariants_ind]
        self.invariants_coef = [x.to(device) for x in self.invariants_coef]

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
        coords = np.linspace(lower, upper, self.cube_side)
        [x, y, z] = np.meshgrid(coords, coords, coords)
        x = x.flatten(order=order)
        y = y.flatten(order=order)
        z = z.flatten(order=order)
        return x, y, z

    def image_size(self) -> int:
        return self.cube_side ** 3

    @abstractmethod
    def init_moments2invariants(self) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        pass

    @abstractmethod
    def init_polynomials(self) -> np.ndarray:
        pass

    @abstractmethod
    def pre_invariant_moments_normalization(self, moments: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def normalization_invariants(self, invariants: torch.Tensor) -> torch.Tensor:
        pass

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
                               inv_idx[..., 2]]
            out[:, idx] = torch.sum(torch.prod(products, dim=-2) * inv_coef, dim=-1)

        # TODO: post_invariant normalization
        return self.normalization_invariants(out)


class AppellInvariant3D(Invariant3D):

    def __init__(self,
                 appell_type: Appell_Type,
                 appell_weight: Appell_polynomial_weights,
                 appell_parameter_s: float,
                 appell_type_s: int,
                 num_invariants: int,
                 cube_side: int,
                 max_rank: int,
                 device: torch.device):
        self.appell_type = appell_type
        self.appell_weight = appell_weight
        # TODO: Make Tensor
        self.appell_parameter_s = appell_parameter_s
        self.appell_parameter_v = torch.tensor(data=2 + (appell_type_s % 2) if appell_type_s > 0 else 3,
                                               dtype=torch.float64,
                                               device=device)
        self.appell_type_s = appell_type_s
        self.normalization_factor = torch.tensor(data=((cube_side - 1) / 2) ** 3,
                                                 dtype=torch.float64,
                                                 device=device)
        # Convert to Tensor
        super().__init__(num_invariants=num_invariants,
                         cube_side=cube_side,
                         max_rank=max_rank,
                         device=device)
        # self.invariants_ind init in super class
        self.invariants_sizes = torch.tensor(
            data=[torch.sum(ind[:, 0, :]) / self.appell_parameter_v for ind in self.invariants_ind],
            dtype=torch.float64,
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

    def _get_matlab_polynomials(self) -> np.ndarray:
        x, y, z = self.get_coords(lower=-1, upper=1, order='F')
        return matlab_appell_polynomials_recursive_3d(self.max_rank, self.max_rank, self.max_rank,
                                                      x, y, z,
                                                      self.appell_type.value,
                                                      self.appell_parameter_s,
                                                      self.appell_weight.value)

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

    def _get_matlab_invariants(self, images: torch.Tensor):
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
                                                      types=self.appell_type_s,
                                                      typeg=1)
        return matlab_invariants

    def pre_invariant_moments_normalization(self, moments: torch.Tensor) -> torch.Tensor:
        if self.appell_type_s > 0:
            # NOTE: normalization is inplace
            return moments_volume_normalization(moments, self.appell_type_s)
        else:
            return moments

    def normalization_moments(self, moments: torch.Tensor):
        return moments / self.normalization_factor

    def normalization_invariants(self, invariants: torch.Tensor) -> torch.Tensor:
        return torch.sign(invariants) * (torch.abs(invariants) ** (1.0 / self.invariants_sizes))


class GauseHermiteInvariants3D(Invariant3D):
    def __init__(self,
                 sigma: float,
                 normcoef: float,
                 normsize, #TODO: Enum
                 num_invariants: int,
                 cube_side: int,
                 max_rank: int,
                 device: torch.device):
        self.sigma = sigma
        self.normcoef = normcoef
        self.normsize = normsize
        # TODO: Add parameters
        super().__init__(num_invariants=num_invariants,
                         cube_side=cube_side,
                         max_rank=max_rank,
                         device=device)
        
    def init_moments2invariants(self) -> Tuple[Sequence[torch.Tensor], Sequence[torch.Tensor]]:
        # TODO: refactors
        return [torch.zeros((1,1,1))], [torch.zeros((1,1,1))]

    def init_polynomials(self) -> np.ndarray:
        # TODO: refactors
        return np.zeros(1)

    def pre_invariant_moments_normalization(self, moments: torch.Tensor) -> torch.Tensor:
        pass

    def normalization_invariants(self, invariants: torch.Tensor) -> torch.Tensor:
        pass

    def normalization_moments(self, moments: torch.Tensor) -> torch.Tensor:
        pass

    def _get_matlab_polynomials(self) -> np.ndarray:
        pass

    def _get_matlab_moments(self, images: torch.Tensor) -> np.ndarray:
        matlab_moments = np.zeros([images.shape[0], self.max_rank + 1, self.max_rank + 1, self.max_rank + 1])
        for idx, image in enumerate(images):
            matlab_moments[idx] = (
                matlab_gauss_hermite_moments(img=image.cpu().numpy(),
                                             rank=self.max_rank,
                                             sigma=self.sigma,
                                             normsize=self.normsize,
                                             normcoef=self.normcoef))

        return matlab_moments

    def _get_matlab_invariants(self, images: torch.Tensor) -> np.ndarray:
        pass
