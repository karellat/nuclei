import os
from loguru import logger
import numpy as np
import torch
from appell_invariant import InvariantAppell
import scipy.io
from dataset import get_mask, get_data

from config import MAX_RANK, SPHERE_RADIUS, model_type, SRZ, INVARIANTS_NUM, model_params, MASK_NUM, PATH, PATH_MASK, \
    PATH_CLASSES
from invariant3d import Invariant3D

# Setting
DEBUG = False
assert torch.cuda.is_available()
device = torch.device('cuda')
if DEBUG:
    from matlab_bridge import matlab_image_to_invariants

    logger.debug("Debug mode")


def _init_model(model, model_params, device) -> Invariant3D:
    return model(**model_params, device=device)


numpy_worm = get_data(PATH)
numpy_masks = get_mask(PATH_MASK)

mask_indicies = np.unique(numpy_masks)
mask_indicies = mask_indicies[np.linspace(10,
                                          mask_indicies.shape[0] - 10,
                                          MASK_NUM - 2).astype(int)]

model = _init_model(model_type, model_params, device)

# TODO: Remove training samples
for mask_idx in mask_indicies:
    mask = numpy_masks == mask_idx
    masked = numpy_worm * mask
    x, y, z = np.where(mask)
    m000 = np.sum(masked)
    m100 = np.sum(x * masked[x, y, z])
    m010 = np.sum(y * masked[x, y, z])
    m001 = np.sum(z * masked[x, y, z])
    cx = int(m100 / m000)
    cy = int(m010 / m000)
    cz = int(m001 / m000)
    cube = torch.from_numpy(masked[
                            cx - SPHERE_RADIUS:cx + SPHERE_RADIUS + 1,
                            cy - SPHERE_RADIUS:cy + SPHERE_RADIUS + 1,
                            cz - SPHERE_RADIUS:cz + SPHERE_RADIUS + 1]).to(device)
    np.testing.assert_allclose(torch.sum(cube).cpu(), np.sum(masked))
    invariants = torch.zeros((1, INVARIANTS_NUM)).to(device)
    invariants = model.invariants(images=cube.reshape((1, SRZ, SRZ, SRZ)),
                                  out=invariants)

    if not os.path.exists(PATH_CLASSES):
        os.mkdir(PATH_CLASSES)

    scipy.io.savemat(os.path.join(PATH_CLASSES, f'nuclei_{mask_idx}.mat'),
                     dict(model=model_type.__name__,
                          **model_params,
                          file=PATH,
                          mask_file=PATH_MASK,
                          rank=MAX_RANK,
                          mask_idx=mask_idx,
                          cube=cube.cpu().numpy(),
                          invariants=invariants.cpu().numpy(),
                          tx=cx,
                          ty=cy,
                          tz=cz))
