from numba import jit
import numpy as np


@jit(nopython=True)
def cos_similatiry(vector1, vector2):
    cos = np.sum(vector1 * vector2) / (np.sqrt(np.sum(vector1 ** 2)) * np.sqrt(np.sum(vector2 ** 2)))
    return cos
