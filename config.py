from appell_polynomials_3D import Appell_Type, Appell_polynomial_weights
from invariant3d import ZernikeInvariants3D, AppellInvariant3D, GaussHermiteInvariants3D, ZernikeMomentsNormalization, GeometricInvariants3D

# Basic params
SPHERE_RADIUS = 20
BATCH_SIZE = 1
MAX_RANK = 6
# First
#MASK_IDS = [107, 156, 204, 253, 302, 351, 399, 448, 497, 58]
# Second
MASK_IDS = [10, 129, 188, 248, 307, 367, 426, 486, 546, 69]
CM = "IMAGE" # MASK - center of mask calculated from image or mask
PATH = "data/c_elegans_nuclei/train/images/C18G1_2L1_1.tif"
PATH_MASK = "data/c_elegans_nuclei/train/masks/C18G1_2L1_1.tif"
NAME = "test"
PATH_CLASSES = f"classes_{NAME}"
OUTPUT_NAME = f"result_{NAME}"
SKIP_ZEROS = True
INVARIANTS_NUM = 77
SRZ = SPHERE_RADIUS * 2 + 1
TYPEG = 1
# Model
model_type = GeometricInvariants3D
# Parameters
model_params = dict(
    typeg=TYPEG,
    # Magnitude normalization for geometric invariants
    types=1,
    num_invariants=INVARIANTS_NUM,
    cube_side=SRZ,
    max_rank=MAX_RANK,
)
