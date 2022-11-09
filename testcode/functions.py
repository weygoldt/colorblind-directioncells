import random

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy import interpolate
from scipy.signal import butter, periodogram, sosfiltfilt
from scipy.stats import gamma, norm


def find_closest(array, target):
    """Takes an array and a target and returns an index for a value of the array that matches the target most closely.

    Could also work with multiple targets and may not work for unsorted arrays, i.e. where a value occurs multiple times. Primarily used for time vectors.

    Parameters
    ----------
    array : array, required
        The array to search in.
    target : float, required
        The number that needs to be found in the array.

    Returns
    ----------
    idx : array,
        Index for the array where the closes value to target is.
    """
    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array) - 1)
    left = array[idx - 1]
    right = array[idx]
    idx -= target - left < right - target
    return idx
