import numpy as np

from termcolors import TermColor as tc


def find_closest(array, target, limit=True, verbose=True):
    """Takes an array and a target and returns an index for a value of the array that matches the target most closely.

    Could also work with multiple targets and may not work for unsorted arrays, i.e. where a value occurs multiple times. Primarily used for time vectors. If limit is enabled, the difference between the target and a value on the array can not be smaller than half
    of the difference between two points on the array.

    Parameters
    ----------
    array : array, required
        The array to search in.
    target : float, required
        The number that needs to be found in the array.
    limit : bool, default True
        To limit or not to limit the difference between target and array value.
    verbose : bool, default True
        To print or not to print warnings.

    Returns
    ----------
    idx : array,
        Index for the array where the closes value to target is.
    """

    # find the closest value
    idx = array.searchsorted(target)
    idx = np.clip(idx, 1, len(array) - 1)
    left = array[idx - 1]
    right = array[idx]
    idx -= target - left < right - target

    # report error if limit is true
    if limit:
        dt = array[1] - array[0]

        # check if difference between index and target is larger than half the sampling dt
        if abs(array[idx]-target) > dt/2:
            idx = np.nan
            if verbose:
                print(
                    f"{tc.err('[ERROR]')} [functions.find_closest] The data point (target) is not on the array!")

    # do the same check but still return the value and just print a warning
    else:
        dt = array[1] - array()[0]
        if abs(array[idx]-target) > dt/2:
            if verbose:
                print(
                    f"{tc.warn('[WARNING]')} [functions.find_closest] The data point (target) is not on the array!")

    return idx

