import os
from loguru import logger
import numpy as np
import torch
import scipy.io
from dataset import get_mask, get_data

from config import MAX_RANK, SPHERE_RADIUS, model_type, CM, SRZ, INVARIANTS_NUM, model_params, MASK_IDS, PATH, PATH_MASK, \
    PATH_CLASSES, NAME
from invariant3d import Invariant3D

# Setting
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def _init_model(model, model_params, device) -> Invariant3D:
    return model(**model_params, device=device)


numpy_worm = get_data(PATH)
logger.debug(f"Loading data from {PATH}, intensity mean {np.mean(numpy_worm)}+-{np.std(numpy_worm)}")
numpy_masks = get_mask(PATH_MASK)

model = _init_model(model_type, model_params, device)
logger.debug(f"Model {type(model).__name__}")
for k, v in model_params.items():
    logger.debug(f"\t{k}:{v}")
logger.debug(f"Preparing training set, CM method: {CM}")
for mask_idx in MASK_IDS:
    logger.debug(f"Calculating idx: {mask_idx}")
    mask = numpy_masks == mask_idx
    masked = numpy_worm * mask
    x, y, z = np.where(mask)
    m000 = np.sum(masked)
    m100 = np.sum(x * masked[x, y, z])
    m010 = np.sum(y * masked[x, y, z])
    m001 = np.sum(z * masked[x, y, z])
    cx_image = int(m100 / m000)
    cy_image = int(m010 / m000)
    cz_image = int(m001 / m000)
    cx_mask = np.mean(x).astype(int)
    cy_mask = np.mean(y).astype(int)
    cz_mask = np.mean(z).astype(int)
    if CM.upper() == "IMAGE":
        cx = cx_image
        cy = cy_image
        cz = cz_image
    elif CM.upper() == "MASK":
        cx = cx_mask
        cy = cy_mask
        cz = cz_mask
    else:
        raise NotImplementedError("Unknown method form calculating center of mass.")
    logger.debug(f"\t center of mass: {(cx,cy,cz)}; difference between methods {(cx_image-cx_mask, cy_image - cy_mask, cz_image - cz_mask)}.")
    cube = torch.from_numpy(masked[
                            cx - SPHERE_RADIUS:cx + SPHERE_RADIUS + 1,
                            cy - SPHERE_RADIUS:cy + SPHERE_RADIUS + 1,
                            cz - SPHERE_RADIUS:cz + SPHERE_RADIUS + 1]).to(device)
    if not np.isclose(torch.sum(cube).cpu(), np.sum(masked)):
        logger.warning(f"Cube idx {mask_idx} is not covering the whole mask") 
    invariants = torch.zeros((1, INVARIANTS_NUM)).to(device)
    invariants = model.invariants(images=cube.reshape((1, SRZ, SRZ, SRZ)),
                                  out=invariants)

    _output_dir = os.path.join('results', NAME, PATH_CLASSES)
    os.makedirs(_output_dir, exist_ok=True)

    _path = os.path.join(_output_dir, f'nuclei_{mask_idx}.mat')
    scipy.io.savemat(_path,
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
    logger.debug(f"\tSaving as {_path}.")
logger.debug(f"Training set done.")
