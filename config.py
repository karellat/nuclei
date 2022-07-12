from appell_polynomials_3D import Appell_Type, Appell_polynomial_weights
from invariant3d import AppellInvariant3D, GaussHermiteInvariants3D

# Basic params
SPHERE_RADIUS = 17
BATCH_SIZE = 10
MAX_RANK = 6
MASK_NUM = 12
PATH = "data/c_elegans_nuclei/train/images/C18G1_2L1_1.tif"
PATH_MASK = "data/c_elegans_nuclei/train/masks/C18G1_2L1_1.tif"
PATH_CLASSES = "classes_gausshermite"
OUTPUT_NAME = "result_gausshermite"
SKIP_ZEROS = True
INVARIANTS_NUM=77
SRZ=SPHERE_RADIUS * 2 + 1
# Model
model_type = GaussHermiteInvariants3D
# Parameters
model_params = dict( 
      types=0,
      sigma=0.3,
      normcoef=0.5,
      num_invariants=INVARIANTS_NUM,
      cube_side=SRZ,
      max_rank=MAX_RANK
)
