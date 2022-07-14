from zipfile import ZipFile
from scipy.io import savemat
import scipy.io
from tqdm import tqdm
import numpy as np
from glob import glob
from loguru import logger
import os
import torch
from dataset import get_data

# Config
from config import SKIP_ZEROS, MASK_IDS, SPHERE_RADIUS, NAME, BATCH_SIZE, model_type, SRZ, INVARIANTS_NUM, model_params,OUTPUT_NAME, PATH, PATH_CLASSES
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

# Invariants
model = _init_model(model_type, model_params, device)
logger.debug("Model initialized.")

# Load training set
classes = torch.zeros([10, 77], dtype=torch.float64).to(device)

_training_files = glob(os.path.join('results', NAME, PATH_CLASSES, '*.mat'))
assert len(_training_files) == len(MASK_IDS)
for idx, f in enumerate(_training_files):
    logger.debug(f"\tLoading {idx}. sample from {f}")
    mat_file = scipy.io.loadmat(f, mat_dtype=True)
    assert mat_file['cube_side'] == model.cube_side
    classes[idx] = torch.from_numpy(mat_file['invariants'])
logger.debug("Training sample invariants loaded.")

# Prepare Data
worm = torch.from_numpy(numpy_worm).to(device)
sphere_mask = torch.from_numpy(get_sphere_mask(SPHERE_RADIUS)).to(device)
invariant_out = torch.zeros((BATCH_SIZE, INVARIANTS_NUM)).to(device)
logger.debug("Data prepared.")

distance_result = -1 * torch.ones([*worm.shape], dtype=torch.float64).to(device)
distance_arg = -1 * torch.ones([*worm.shape], dtype=torch.int64).to(device)



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
    _output_dir = os.path.join('results', NAME)
    torch.save(distance_result, os.path.join(_output_dir, f'{OUTPUT_NAME}_distance.pt'))
    torch.save(distance_arg, os.path.join(_output_dir, f'{OUTPUT_NAME}_argmin.pt'))
pbar_x.close()


logger.debug("Calculation finished.")
# Writing output
distances = torch.load(os.path.join('results', NAME, f'result_{NAME}_distance.pt'),
                       map_location=torch.device('cpu'))
indexes = torch.load(os.path.join('results', NAME, f'result_{NAME}_argmin.pt'),
                     map_location=torch.device('cpu'))

matfile_path = f'results/{NAME}/results_{NAME}.mat'
savemat(matfile_path, dict(minimal_distances=distances.numpy(),
                           argmin_distances=indexes.numpy()))

matfile_paths = [*glob(os.path.join('results', NAME, PATH_CLASSES, '*'))]

with ZipFile(os.path.join('results', NAME, f'result_{NAME}.zip'), 'w') as zip_file:
    zip_file.write(matfile_path, arcname=f'results_{NAME}.mat')
    for file in matfile_paths:
        zip_file.write(file, arcname=f'{PATH_CLASSES}/{os.path.basename(file)}')
