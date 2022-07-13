from appell_polynomials_3D import Appell_Type, Appell_polynomial_weights
from invariant3d import AppellInvariant3D, GaussHermiteInvariants3D

# Basic params
SPHERE_RADIUS = 17
BATCH_SIZE = 5
MAX_RANK = 6
MASK_NUM = 12
PATH = "data/c_elegans_nuclei/train/images/C18G1_2L1_1.tif"
PATH_MASK = "data/c_elegans_nuclei/train/masks/C18G1_2L1_1.tif"
NAME = "test"
PATH_CLASSES = f"classes_{NAME}"
OUTPUT_NAME = f"result_{NAME}"
SKIP_ZEROS = True
INVARIANTS_NUM = 77
SRZ = SPHERE_RADIUS * 2 + 1
# Model
model_type = AppellInvariant3D
# Parameters
model_params = dict(
    appell_type=Appell_Type.U,
    appell_weight=Appell_polynomial_weights.WEIGHTED_5,
    appell_parameter_s=1,
    appell_type_s=0,
    num_invariants=INVARIANTS_NUM,
    cube_side=SRZ,
    max_rank=MAX_RANK,
)
