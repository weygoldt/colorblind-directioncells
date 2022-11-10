import numpy as np

from termcolors import TermColor as tc


def find_on_time(array, target, limit=True, verbose=True):
    """Takes a time array and a target (e.g. timestamp) and returns an index for a value of the array that matches the target most closely.

    The time array must (a) contain unique values and (b) must be sorted from smallest to largest. If limit is True, the function checks for each target, if the difference between the target and the closest time on the time array is not larger than half of the distance between two time points at that place. When the distance exceed half the delta t, an error is returned. This also means that the time array must not nessecarily have a constant delta t.

    Parameters
    ----------
    array : array, required
        The array to search in, must be sorted.
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

    def find_closest(array, target):
        idx = array.searchsorted(target)
        idx = np.clip(idx, 1, len(array) - 1)
        left = array[idx - 1]
        right = array[idx]
        idx -= target - left < right - target

        return idx

    def error(verbose):
        if verbose:
            raise ValueError(
                f"{tc.err('[ERROR]')} [functions.find_closest] The data point (target) is not on the array!")

    def warning(verbose):
        if verbose:
            print(
                f"{tc.warn('[WARNING]')} [functions.find_closest] The data point (target) is not on the array!")

    # find the closest value
    idx = find_closest(array, target)

    # compute dt at this point
    found = array[idx]
    dt_target = target - found

    # check if target between first and last value
    if target <= array[-1] and target >= array[0]:
        if dt_target > 0:
            dt_sampled = array[idx+1]-array[idx]
        else:
            dt_sampled = array[idx]-array[idx-1]

        if abs(array[idx]-target) > dt_sampled/2:
            if limit:
                idx = np.nan
                error(verbose)
            else:
                warning(verbose)

    # check if target smaller than first value
    elif target < array[0]:
        dt_sampled = array[1]-array[0]

        if abs(array[idx]-target) > dt_sampled/2:
            if limit:
                idx = np.nan
                error(verbose)
            else:
                warning(verbose)

    # check if target larger than last value
    elif target > array[-1]:
        dt_sampled = array[-1]-array[-2]

        if abs(array[idx]-target) > dt_sampled/2:
            if limit:
                idx = np.nan
                error(verbose)
            else:
                warning(verbose)

    return idx
