import scipy.io
from tqdm import tqdm
import numpy as np
from glob import glob
from loguru import logger
import os
import torch
from appell_invariant import InvariantAppell
from dataset import get_data

# Config
from config import SKIP_ZEROS, SPHERE_RADIUS,BATCH_SIZE, model_type, SRZ, INVARIANTS_NUM, model_params,OUTPUT_NAME, PATH, PATH_CLASSES
from invariant3d import Invariant3D

# Setting
assert torch.cuda.is_available()
device = torch.device('cuda')
logger.debug(f"Running on {device}")


def _init_model(model, model_params, device) -> Invariant3D:
    return model(**model_params, device=device)


def get_sphere_mask(radius):
    srz = radius * 2 + 1
    assert srz % 2 == 1, "Dimension has to be odd"
    middle = int((srz - 1) / 2)
    xs, ys, zs = np.meshgrid(np.arange(srz), np.arange(srz), np.arange(srz))
    xs = (xs - middle) ** 2
    ys = (ys - middle) ** 2
    zs = (zs - middle) ** 2
    dist = xs + ys + zs
    return dist <= (radius ** 2)


numpy_worm = get_data(PATH)

# Load training set

classes = torch.zeros([10, 77], dtype=torch.float64).to(device)

for idx, f in enumerate(glob(os.path.join(PATH_CLASSES, '*.mat'))):
    classes[idx] = torch.from_numpy(scipy.io.loadmat(f, mat_dtype=True)['invariants'])
logger.debug("Training sample invariants loaded.")

# Prepare Data
worm = torch.from_numpy(numpy_worm).to(device)
sphere_mask = torch.from_numpy(get_sphere_mask(SPHERE_RADIUS)).to(device)
invariant_out = torch.zeros((BATCH_SIZE, INVARIANTS_NUM)).to(device)
logger.debug("Data prepared.")

distance_result = -1 * torch.ones([*worm.shape], dtype=torch.float64).to(device)
distance_arg = -1 * torch.ones([*worm.shape], dtype=torch.int64).to(device)


# Invariants
model = _init_model(model_type, model_params, device)


logger.debug("Invariants calculation.")
cache = torch.zeros([BATCH_SIZE, SRZ, SRZ, SRZ]).to(device)
cache_indicies = torch.zeros([BATCH_SIZE, 3], dtype=torch.int64)
pbar_x = tqdm(total=worm.shape[0] - SPHERE_RADIUS, position=0, desc="X:", leave=False, colour='green', ncols=80)
for x in np.arange(SPHERE_RADIUS, worm.shape[0] - SPHERE_RADIUS):
    pbar_y = tqdm(total=worm.shape[1] - SPHERE_RADIUS,  position=1, desc="Y:", leave=False, colour='red', ncols=80)
    for y in np.arange(SPHERE_RADIUS, worm.shape[1] - SPHERE_RADIUS):
        cache_idx = 0
        for z in np.arange(SPHERE_RADIUS, worm.shape[2] - SPHERE_RADIUS):
            cube = (
                worm[x - SPHERE_RADIUS:x + SPHERE_RADIUS + 1,
                     y - SPHERE_RADIUS:y + SPHERE_RADIUS + 1,
                     z - SPHERE_RADIUS:z + SPHERE_RADIUS + 1]
                   )
            if SKIP_ZEROS and (torch.sum(cube) == 0):
                distance_result[x, y, z] = -2
                distance_arg[x, y, z] = -2
                continue
            cache[cache_idx] = cube
            cache_indicies[cache_idx, 0] = x
            cache_indicies[cache_idx, 1] = y
            cache_indicies[cache_idx, 2] = z
            if cache_idx == (BATCH_SIZE - 1) or z == (worm.shape[2] - SPHERE_RADIUS - 1):
                invariant_out = model.invariants(cache * sphere_mask, invariant_out)
                distance = torch.cdist(invariant_out[:cache_idx+1], classes)
                min_distance, argmin_distance = torch.min(distance, dim=-1)
                distance_result[cache_indicies[:cache_idx+1, 0],
                                cache_indicies[:cache_idx+1, 1],
                                cache_indicies[:cache_idx+1, 2]] = min_distance
                distance_arg[cache_indicies[:cache_idx+1, 0],
                             cache_indicies[:cache_idx+1, 1],
                             cache_indicies[:cache_idx+1, 2]] = argmin_distance

                # Null cache
                cache_idx = 0
            else:
                cache_idx += 1
        pbar_y.update(1)
    pbar_y.close()
    pbar_x.update(1)
    # Save semi-result
    torch.save(distance_result, f'{OUTPUT_NAME}_distance.pt')
    torch.save(distance_arg, f'{OUTPUT_NAME}_argmin.pt')
pbar_x.close()

logger.debug("Calculation finished.")