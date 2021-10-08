import numpy as np


def find_nearest(array, value):
    """Returns the index of the value closest to `value` in the array"""
    array = np.asarray(array)
    idx = (np.abs(array - value)).argmin()
    return idx
