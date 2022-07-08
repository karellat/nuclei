from appell_polynomials_3D import Appell_Type, Appell_polynomial_weights

# Parameters
BATCH_SIZE = 10
MAX_RANK = 6
SPHERE_RADIUS = 17
APPELL_TYPE = Appell_Type.U
APPELL_PARAM = 1
APPELL_WEIGHT = Appell_polynomial_weights.WEIGHTED_5
SRZ = SPHERE_RADIUS * 2 + 1
TYPES = 0
INVARIANTS_NUM = 77
MASK_NUM = 12
PATH = "data/c_elegans_nuclei/train/images/C18G1_2L1_1.tif"
PATH_MASK = "data/c_elegans_nuclei/train/masks/C18G1_2L1_1.tif"
PATH_CLASSES = "classes_17"
OUTPUT_NAME = "result_17srz"
