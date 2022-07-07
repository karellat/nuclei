import numpy as np
import torch
from appell_polynomials_3D import appell_polynomials_recursive_3d, Appell_Type, Appell_polynomial_weights, \
    appell_moments_3d_predef, cafmi3d, moments_volume_normalization
from appell_invariant import InvariantAppell
import scipy.io
from dataset import get_mask, get_data
from matlab_bridge import matlab_image_to_invariants

from config import MAX_RANK, SPHERE_RADIUS, APPELL_TYPE, APPELL_PARAM, APPELL_WEIGHT, SRZ, TYPES, INVARIANTS_NUM, MASK_NUM, PATH, PATH_MASK

# Setting
assert torch.cuda.is_available()
device = torch.device('cuda')

numpy_worm = get_data(PATH)
numpy_masks = get_mask(PATH_MASK)

mask_indicies = np.unique(numpy_masks)
mask_indicies = mask_indicies[np.linspace(10,
                                          mask_indicies.shape[0]-10,
                                          MASK_NUM).astype(int)]

model = InvariantAppell(rank=MAX_RANK,
                        appell_type=APPELL_TYPE,
                        appell_weight=APPELL_WEIGHT,
                        appell_param=APPELL_PARAM,
                        invariants_num=INVARIANTS_NUM,
                        types=TYPES,
                        img_srz=SRZ,
                        device=device)

# TODO: Remove training samples
for mask_idx in mask_indicies:
    mask = numpy_masks == mask_idx
    masked = numpy_worm * mask
    x, y, z = np.where(mask)
    cx = np.mean(x).astype(int)
    cy = np.mean(y).astype(int)
    cz = np.mean(z).astype(int)
    cube = torch.from_numpy(masked[
                            cx - SPHERE_RADIUS:cx + SPHERE_RADIUS + 1,
                            cy - SPHERE_RADIUS:cy + SPHERE_RADIUS + 1,
                            cz - SPHERE_RADIUS:cz + SPHERE_RADIUS + 1]).to(device)
    np.testing.assert_allclose(torch.sum(cube).cpu(), np.sum(masked))
    invariants = torch.zeros((1, INVARIANTS_NUM)).to(device)
    # TODO: Assert with the matlab
    matlab_invariant = matlab_image_to_invariants(img=cube.cpu().numpy(),
                                                  srz=SRZ,
                                                  types=TYPES,
                                                  appell_type=APPELL_TYPE,
                                                  appell_param=APPELL_PARAM,
                                                  appell_rank=MAX_RANK,
                                                  appell_weight=APPELL_WEIGHT
                                                  )
    invariants = model.calc_invariants(cube.reshape((1, SRZ, SRZ, SRZ)), invariants)
    np.testing.assert_allclose(matlab_invariant, invariants.cpu().numpy())

    scipy.io.savemat(f'nuclei_{mask_idx}.mat',
                     dict(file=PATH,
                          mask_file=PATH_MASK,
                          rank=MAX_RANK,
                          appell_type=APPELL_TYPE,
                          appell_weight=APPELL_WEIGHT,
                          appell_param=APPELL_PARAM,
                          invariants_num=INVARIANTS_NUM,
                          types=TYPES,
                          img_srz=SRZ,
                          mask_idx=mask_idx,
                          cube=cube.cpu().numpy(),
                          invariants=invariants.cpu().numpy(),
                          tx=cx,
                          ty=cy,
                          tz=cz))

