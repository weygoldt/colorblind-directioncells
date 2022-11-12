from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy.stats import spearmanr
from sklearn.metrics import auc
from sklearn.neighbors import KernelDensity
from tqdm import tqdm

from termcolors import TermColor as tc


def find_right_tail(x, y, target_auc, plot=True):

    # right tail area under curve
    gradient = np.zeros_like(x[:-1])
    index = np.arange(len(x[:-1]))

    # compute difference gradient (between actual auc and target auc)
    for i, t in enumerate(x[:-1]):
        area = auc(x[i:], y[i:])
        gradient[i] = abs(area-target_auc)

    idx = index[gradient == gradient.min()][0]

    if plot:
        fig, ax = plt.subplots(1, 2)

        ax[0].plot(x, y, c="k")
        ax[0].fill_between(x[idx:], np.zeros_like(x[idx:]),
                           y[idx:], zorder=-10, alpha=0.8, facecolor="r")
        ax[0].set_title("Input data and \n marked area")

        ax[1].plot(x[:-1], gradient, c="k")
        ax[1].scatter(x[idx], gradient[idx], c="r", zorder=10)
        ax[1].set_title("Difference gradient \n and lowest point")

        plt.show()

    return gradient, idx


def kde1d(y, bandwidth, xlims="auto", resolution=500, kernel="gaussian"):
    """
    Estimate the probability density function of a continuous variable using the kernel density method.
    Computes the limits by minimum and maximum of the supplied variable or set the maxima (e.g. for variables that
    cannot the smaller then 0 etc.).

    Parameters
    ----------
    y : 1d array-like
        The continuous variable
    bandwidth : float
        The bandwidth of the kernel (i.e. sigma of a gaussian)
    xlims : {array-like, 'auto'}, optional
        The limits of the data, by default "auto"
    resolution : int, optional
        The number of points to draw the KDE, by default 500
    kernel : {'gaussian', 'tophat', 'epanechnikov', 'exponential', 'linear', 'cosine'}
        The kernel to use, by default "gaussian"

    Returns
    -------
    x : array
        The x axis of the estimated PDF.
    pdf : array
        The KDE-estimated PDF of the input data.
    """
    if xlims == "auto":
        x = np.linspace(np.min(y), np.max(y), resolution)
    else:
        try:
            x = np.linspace(xlims[0], xlims[1], resolution)
        except ValueError:
            print("Invalid argument for 'xlims'. Must be a list/array or 'auto'.")
    kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(y[:, None])
    log_dens = kde.score_samples(x[:, None])

    print(
        "computed AreaUnderCurve (AUC) of KDE using sklearn.metrics.auc: {}".format(
            auc(x, np.exp(log_dens))
        )
    )
    return x, np.exp(log_dens)


def flatten(l):
    """
    Flattens a list / array of lists.

    Parameters
    ----------
    l : array or list of lists
        The list to be flattened

    Returns
    -------
    list
        The flattened list
    """
    return [item for sublist in l for item in sublist]


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

    # new shit starts here
    if target <= array[0]:
        dt_sampled = array[idx+1]-array[idx]

        if abs(array[idx]-target) > dt_sampled/2:
            if limit:
                idx = np.nan
                error(verbose)
            else:
                warning(verbose)

    if target > array[0] and target < array[-1]:
        if dt_target >= 0:
            dt_sampled = array[idx+1]-array[idx]
        else:
            dt_sampled = array[idx]-array[idx-1]

        if abs(array[idx]-target) > dt_sampled/2:
            if limit:
                idx = np.nan
                error(verbose)
            else:
                warning(verbose)

    if target >= array[-1]:
        dt_sampled = array[idx] - array[idx-1]

        if abs(array[idx]-target) > dt_sampled/2:
            if limit:
                idx = np.nan
                error(verbose)
            else:
                warning(verbose)

    # and ends here

    """
    # check if target between first and last value
    if target < array[-1] and target >= array[0]:
        if dt_target >= 0:
            dt_sampled = array[idx+1]-array[idx]
        else:
            dt_sampled = array[idx]-array[idx-1]

        if abs(array[idx]-target) > dt_sampled/2:
            if limit:
                idx = np.nan
                error(verbose)
            else:
                warning(verbose)

    if target == array[-1]:
        if dt_target >= 0:
            dt_sampled = array[idx]-array[idx-1]
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
    """

    return idx


def data_one_rec_id(summaryfile, rec_id):
    """Get all rois of one recording

    Parameters
    ----------
    summaryfile : hdf5
        summaryfile with all recordings
    rec_id : int
        chooses wich recording one wants

    Returns
    -------
    list of hdf5
        list of all the rois within one recording
    """
    rois = summaryfile.rois()
    roi_one_rec = []
    for r in rois:
        if r.rec_id == rec_id:
            roi_one_rec.append(r)
    return roi_one_rec


def get_attributes(one_recording):
    """get attributes of one recording.

    Parameters
    ----------
    one_recording : list of vxtools.summarize.structure.Roi
        hdf5 SummaryFile with all rois of the same recording id

    Returns
    -------
    start_time np.array()
        sorted start_times of the stimulus
    stop_time np.array()
        sorted stop_times of the stimulus
    angul_vel list()
        angular velocitiy used in the stimulus
    angul_pre list()
        angular period used for the stimulus
    rgb1 np.array of arrays with 3 colors
        colors used for the first stripe
    rgb2 np.array of arrays with 3 colors
        colors used for the second stripe
    """

    roi = 1
    start_time = []
    end_time = []
    angul_vel = []
    angul_pre = []
    rgb_1 = []
    rgb_2 = []

    array_phases = np.sort(
        one_recording[roi].rec.h5group['display_data']['phases'])
    int_phases = np.sort([int(x) for x in array_phases])

    for ph in int_phases:
        start_time.append(
            one_recording[roi].rec.h5group['display_data']['phases'][f'{ph}'].attrs['start_time'])
        end_time.append(
            one_recording[roi].rec.h5group['display_data']['phases'][f'{ph}'].attrs['end_time'])
        if ph == int_phases[0] or ph == int_phases[-1]:
            angul_vel.append(np.nan)
            angul_pre.append(np.nan)
            rgb_1.append(np.nan)
            rgb_2.append(np.nan)
            continue
        angul_vel.append(one_recording[roi].rec.h5group['display_data']
                         ['phases'][f'{ph}'].attrs['angular_velocity'])

        angul_pre.append(one_recording[roi].rec.h5group['display_data']
                         ['phases'][f'{ph}'].attrs['angular_period'])
        rgb_1.append(
            one_recording[roi].rec.h5group['display_data']['phases'][f'{ph}'].attrs['rgb01'])
        rgb_2.append(
            one_recording[roi].rec.h5group['display_data']['phases'][f'{ph}'].attrs['rgb02'])

    return start_time, end_time, angul_vel, angul_pre, rgb_1, rgb_2


def mean_zscore_roi(one_recording, roi):
    """Calculates the mean zscore between the start and end point of the stimulus of one single ROI

    Parameters
    ----------
    one_recording : list of vxtools.summarize.structure.Roi
        hdf5 SummaryFile with all rois of the same recording id
    roi: int
        index of one ROI of the list one recording 

    Returns
    -------
    list
       mean zscore for evry start and stop time
    """
    start_time, end_time, angul_vel, angul_pre, rgb_1, rgb_2 = get_attributes(
        one_recording)
    times = one_recording[0].times

    time_snippets = []

    for st, end in zip(start_time, end_time):
        start_inx = find_on_time(times, st)
        end_inx = find_on_time(times, end)
        time_snippets.append(np.arange(start_inx, end_inx))

    mean_zscore = []
    for snip in time_snippets:
        zscores_snip = one_recording[roi].zscore[snip]
        mean_zscore.append(np.mean(zscores_snip))

    return mean_zscore


def get_mean_dffs(roi_dffs, times, startstop):

    snippet_indices = []
    center_indices = []
    for st, end in zip(startstop[0], startstop[1]):
        start_inx = find_on_time(times, st)
        end_inx = find_on_time(times, end)
        center_inx = find_on_time(times, st + (end-st)/2)
        center_indices.append(center_inx)
        snippet_indices.append(np.arange(start_inx, end_inx))

    mean_dffs = np.empty((len(roi_dffs[:, 0]), len(snippet_indices)))
    for i in range(len(roi_dffs[:, 0])):
        roi = roi_dffs[i, :]
        mean_dff = np.array([np.mean(roi[snip])
                            for snip in snippet_indices])
        mean_dffs[i, :] = mean_dff

    return mean_dffs, times[center_indices]


def mean_dff_roi(one_recording, roi):
    """Calculates the mean dff between the start and end point of the stimulus of one single ROI

    Parameters
    ----------
    one_recording : list of vxtools.summarize.structure.Roi
        hdf5 SummaryFile with all rois of the same recording id
    roi: int
        index of one ROI of the list one recording 

    Returns
    -------
    list
       mean zscore for evry start and stop time
    """
    start_time, end_time, angul_vel, angul_pre, rgb_1, rgb_2 = get_attributes(
        one_recording)
    times = one_recording[0].times
    time_snippets = []
    for st, end in zip(start_time, end_time):
        start_inx = find_on_time(times, st)
        end_inx = find_on_time(times, end)
        time_snippets.append(np.arange(start_inx, end_inx))

    mean_dff = []
    for snip in time_snippets:
        zscores_snip = one_recording[roi].dff[snip]
        mean_dff.append(np.mean(zscores_snip))

    return mean_dff


def repeats(startstop, nrepeats=3):

    # check if starts and stops are the same lenght
    if len(startstop[0]) != len(startstop[1]):
        raise ValueError(
            f'{tc.err("ERROR")} [ functions.repeats ] The starts and stops are not the same length!')

    # get starts and stops
    starts = startstop[0]
    stops = startstop[1]
    indices = np.arange(len(starts))
    frac = len(indices)/nrepeats

    # check if array lengths are dividable by nrepeats
    if frac % 1 != 0:
        raise ValueError(
            f'{tc.err("ERROR")} [ functions.repeats ] Cant divide by {nrepeats}!')

    repeat_starts = np.empty(nrepeats)
    repeat_stops = np.empty(nrepeats)

    # get starts and stops
    for i in range(nrepeats):
        repeat_starts[i] = i*frac
        repeat_stops[i] = np.arange(i*frac, i*frac+frac)[-1]
    # reshape
    idxs = np.array([np.array([int(x), int(y)], dtype=int)
                    for x, y in zip(repeat_starts, repeat_stops)], dtype=int)

    return idxs


"""
nonnan_angul_vel = [x for x in angul_vel if np.isnan(x) == False]

nr = len(nonnan_angul_vel)/nrepeats
if nr % 1 != 0:
    raise ValueError(f'Cant divide by {nrepeats}!')
inx = []
for i in np.arange(1, nrepeats+1):
    if i == 1:
        inx.append((0, nr))
    else:
        inx.append(((i-1)*nr, nr*i))
inx = np.array(inx, dtype=int)
return inx
"""


def corr_repeats(mean_score, inx):
    """Calculate the the correlation of one ROI within nrepeats

    Parameters
    ----------
    mean_score : list or 1darray
        mean_score dff or zscore values in stimulus windows 
    inx : tupel
        index tupel, where the repeats of one recording starts and stops 

    Returns
    -------
    ndarray
        mean value of three correlations for one ROI 
    """
    z = []
    for i in inx:
        z.append(mean_score[int(i[0]):int(i[-1])])
    combs = [comb for comb in combinations(z, 2)]
    spear = []
    for co in combs:
        spear.append(spearmanr(co[0], co[1])[0])

    spear_mean = np.mean(spear)

    return spear_mean


def sort_rois(mean_dffs, inx):
    """calculate all active ROIs with a threshold. 
    ROIs who have a high correlation with themselfs over time, are active rois 
    Parameters
    ----------
    one_recording : list of vxtools.summarize.structure.Roi
        hdf5 SummaryFile with all rois of the same recording id

    inx : tupel
        index tupel, where the repeats of one recording starts and stops 

    threshold : float, optional
        threshold of the correlation factor, by default 0.6

    Returns
    -------
    2ndarray
        1.dimension are the index fot the ROIs 
        2.dimension are sorted correlation factors
    """
    spearmeans = []
    for i in tqdm(range(len(mean_dffs[:, 0]))):

        # start_time = time.time()
        means = mean_dffs[i, :]
        # print("--- %s seconds ---" % (time.time() - start_time))

        # start_time = time.time()
        spear_mean = corr_repeats(means, inx)
        # print("--- %s seconds ---" % (time.time() - start_time))

        spearmeans.append(spear_mean)

    # index = np.arange(len(spearmeans), dtype=int)
    # active_roi = index[np.array(spearmeans) >= threshold]
    # pearmeans_a_roi = np.array(spearmeans)[active_roi]

    result = np.empty((len(mean_dffs[:, 0]), 2))
    result[:, 0] = np.arange(len(mean_dffs[:, 0]))
    result[:, 1] = spearmeans
    active_rois_sorted = np.array(
        sorted(result, key=lambda x: x[1], reverse=True))

    return active_rois_sorted


def thresh_correlations(sorted_rois, threshold):
    index = np.arange(len(sorted_rois[:, 0]))
    thresh_index = index[abs(sorted_rois[:, 1]) > threshold]
    thresh_rois = sorted_rois[thresh_index, :]

    return thresh_rois


def plot_ang_velocity(onerecording, angul_vel, rois):
    """plot the boxplot of rois with regards to the stimulus angular Velocity 

    Parameters
    ----------
    one_recording : list of vxtools.summarize.structure.Roi
        hdf5 SummaryFile with all rois of the same recording id
    roi: int
        index of one ROI of the list one recording 
    """
    for r in rois:
        mean_zscores = mean_dff_roi(onerecording, r)
        z_vel_30 = [z for z, a in zip(mean_zscores, angul_vel) if a == 30.0]
        z_vel_minus30 = [z for z, a in zip(
            mean_zscores, angul_vel) if a == -30.0]
        z_vel_0 = [z for z, a in zip(mean_zscores, angul_vel) if a == 0.0]

        fig, ax = plt.subplots()
        ax.boxplot(z_vel_0, positions=[1])
        ax.boxplot(z_vel_minus30, positions=[2])
        ax.boxplot(z_vel_30, positions=[3])
        ax.set_xticklabels(['0vel', '-30vel', '30vel'])
        plt.show()


def meanstack(roi_dffs, times, inx):

    meanstack_times = times[inx[0][0]:inx[0][1]]
    meanstack_dffs = np.empty((len(roi_dffs[:, 0]), inx[0][1]))

    for roi in range(len(roi_dffs[:, 0])):
        dff = roi_dffs[roi, :]
        split_dff = np.array([dff[x[0]:x[1]] for x in inx])
        mean_dff = np.mean(split_dff, axis=0)
        meanstack_dffs[roi, :] = mean_dff

    return meanstack_times, meanstack_dffs
