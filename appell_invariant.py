import numpy as np
import torch

from appell_polynomials_3D import moments_volume_normalization, cafmi3d_torch, appell_polynomials_recursive_3d, appell_moments_3d_predef


class InvariantAppell:
    def __init__(self,
                 rank,
                 appell_type,
                 appell_weight,
                 appell_param,
                 invariants_num,
                 types,
                 img_srz,
                 device):
        self.device = device
        # Polynomial
        coords = np.linspace(-1, 1, img_srz)
        [x, y, z] = np.meshgrid(coords, coords, coords)
        x = x.flatten(order='F')
        y = y.flatten(order='F')
        z = z.flatten(order='F')

        _polynom = appell_polynomials_recursive_3d(rank, rank, rank,
                                                   x, y, z,
                                                   appell_type=appell_type,
                                                   s=appell_param,
                                                   weight=appell_weight)
        self.polynomials = torch.from_numpy(_polynom
            .reshape((rank+1, rank+1, rank+1, img_srz, img_srz, img_srz), order='F')
            .reshape(rank+1, rank+1, rank+1, img_srz ** 3)
        ).to(device)

        # Moments to invariants
        self.types = types
        self.v = 2 + (types % 2) if types > 0 else 3

        # Prepare invariants parameters
        moments2invariants = np.load("torch_invariants3Dinv10/moments2invariants.npz",
                                     allow_pickle=True)
        invariant_ind = list(moments2invariants['ind'][0][:invariants_num])
        invariant_coef = list(moments2invariants['coef'][0][0][:invariants_num])
        invariant_sizes = np.array([np.sum(ind[:, 0, :]) / self.v for ind in invariant_ind])

        # Test uint8
        for ind in invariant_ind:
            assert np.max(ind) <= 255

        # Convert to Tensor
        self.invariant_coef = [torch.from_numpy(np.array(coef, dtype=np.float64)).to(device) for coef in invariant_coef]
        self.invariant_ind = [ind.astype(int) for ind in invariant_ind]
        self.invariant_sizes = torch.from_numpy(invariant_sizes).to(device)

    def calc_invariants(self, imgs: torch.Tensor, out: torch.Tensor):
        min_img_side = np.min(imgs[0].shape)
        moments = appell_moments_3d_predef(imgs, self.polynomials, min_img_side)

        if self.types > 0:
            # NOTE: normalization is inplace
            moments = moments_volume_normalization(moments, self.types)

        return cafmi3d_torch(moments=moments,
                             invariant_ind=self.invariant_ind,
                             invariant_coef=self.invariant_coef,
                             invariant_sizes=self.invariant_sizes,
                             out=out)
