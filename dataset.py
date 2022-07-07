import numpy as np
from skimage import io


def get_data(path, normalization=True):
    numpy_worm = io.imread(path)
    # TODO: Fix worm size
    assert numpy_worm.dtype == np.uint8
    if normalization:
        numpy_worm = numpy_worm / 255.0
    return numpy_worm


def get_mask(path):
    return io.imread(path)

