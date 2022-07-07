import numpy as np
import matplotlib.pyplot as plt
import scipy.io
from tqdm import tqdm
import numpy as np
from glob import glob
import torch
from PIL import Image
from skimage import io
import torch
from appell_polynomials_3D import appell_polynomials_recursive_3d, Appell_Type, Appell_polynomial_weights, \
    appell_moments_3d_predef, cafmi3d, moments_volume_normalization
from appell_invariant import InvariantAppell
from dataset import get_data

# Config
from config import BATCH_SIZE, MAX_RANK, SPHERE_RADIUS, APPELL_TYPE, APPELL_PARAM, APPELL_WEIGHT, SRZ, TYPES, INVARIANTS_NUM, PATH

# Setting
assert torch.cuda.is_available()
device = torch.device('cuda')


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

for idx, f in enumerate(glob('classes/*.mat')):
    classes[idx] = torch.from_numpy(scipy.io.loadmat(f, mat_dtype=True)['invariants'])

# Prepare Data
worm = torch.from_numpy(numpy_worm).to(device)
sphere_mask = torch.from_numpy(get_sphere_mask(SPHERE_RADIUS)).to(device)
invariant_out = torch.zeros((BATCH_SIZE, INVARIANTS_NUM)).to(device)

distance_result = -1 * torch.ones([*worm.shape], dtype=torch.float64).to(device)
distance_arg = -1 * torch.ones([*worm.shape], dtype=torch.int64).to(device)


# Invariants
model = InvariantAppell(rank=MAX_RANK,
                        appell_type=APPELL_TYPE,
                        appell_weight=APPELL_WEIGHT,
                        appell_param=APPELL_PARAM,
                        invariants_num=INVARIANTS_NUM,
                        types=TYPES,
                        img_srz=SRZ,
                        device=device)


cache = torch.zeros([BATCH_SIZE, SRZ, SRZ, SRZ]).to(device)
cache_indicies = torch.zeros([BATCH_SIZE, 3], dtype=torch.int64)
pbar_x = tqdm(total=worm.shape[0] - SPHERE_RADIUS, position=0, desc="X:", leave=False, colour='green', ncols=80)
for x in np.arange(SPHERE_RADIUS, worm.shape[0] - SPHERE_RADIUS):
    pbar_y = tqdm(total=worm.shape[1] - SPHERE_RADIUS,  position=1, desc="Y:", leave=False, colour='red', ncols=80)
    for y in np.arange(SPHERE_RADIUS, worm.shape[1] - SPHERE_RADIUS):
        cache_idx = 0
        for z in np.arange(SPHERE_RADIUS, worm.shape[2] - SPHERE_RADIUS):
            cache[cache_idx] = (
                worm[x - SPHERE_RADIUS:x + SPHERE_RADIUS + 1, y - SPHERE_RADIUS:y + SPHERE_RADIUS + 1,
                z - SPHERE_RADIUS:z + SPHERE_RADIUS + 1]
            )
            cache_indicies[cache_idx, 0] = x
            cache_indicies[cache_idx, 1] = y
            cache_indicies[cache_idx, 2] = z
            if cache_idx == 9 or z == (worm.shape[2] - SPHERE_RADIUS - 1):
                invariant_out = model.calc_invariants(cache * sphere_mask, invariant_out)
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
    torch.save(distance_result, 'distance_result.pt')
    torch.save(distance_arg, 'distance_arg.pt')
pbar_x.close()