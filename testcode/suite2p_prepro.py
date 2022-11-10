import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy.stats import spearmanr
from tqdm import tqdm
from vxtools.summarize.structure import SummaryFile

from functions import find_on_time as fc

f = SummaryFile('../data/Summary.hdf5')
# get all rois
# get one recording


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
    one_recording : vxtools.summarize.structure.Roi
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
    range_phases = np.arange(int_phases[0], int_phases[-1])

    for ph in range_phases:
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
    one_recording : vxtools.summarize.structure.Roi
        hdf5 SummaryFile with all rois of the same recording id

    Returns
    -------
    list
       mean zscore for evry start and stop time
    """
    start_time, end_time, angul_vel, angul_pre, rgb_1, rgb_2 = get_attributes(
        one_recording)

    time_snippets = []
    for st, end in zip(start_time, end_time):
        start_inx = fc(time, st)
        end_inx = fc(time, end)
        time_snippets.append(np.arange(start_inx, end_inx))

    mean_zscore = []
    for snip in time_snippets:
        zscores_snip = one_recording[roi].zscore[snip]
        mean_zscore.append(np.mean(zscores_snip))

    return mean_zscore


one_rec = data_one_rec_id(f, 5)
time = one_rec[0].times
start_time, end_time, angul_vel, angul_pre, rgb_1, rgb_2 = get_attributes(
    one_rec)

rposs = np.zeros(2, len())
rnegs = np.zeros(2, len())
for roi in tqdm(range(len(one_rec))):

    # make logical arrays for left and right turning motion
    mean_zscore = mean_zscore_roi(one_rec, roi)[1:-1]
    angveloc_pos = [1 if a == 30.0 else 0 for a in angul_vel][1:-1]
    angveloc_neg = [1 if a == -30.0 else 0 for a in angul_vel][1:-1]

    # compute correlation
    rposs = spearmanr(mean_zscore, angveloc_pos)
    rnegs = spearmanr(mean_zscore, angveloc_neg)


for i, s in enumerate(rposs):
    plt.scatter(i, s[0])


"""
# convert to numpy array
rposs = np.array(rposs)
rnegs = np.array(rnegs)

# threshold
rposs[rposs < 0.3] = 0
rnegs[rnegs < 0.3] = 0

# threshold by significance

# find indices, which are rois
index_array = np.arange(len(rposs))
pos_rois = index_array[rposs != 0]
neg_rois = index_array[rnegs != 0]


# plot
rois = pos_rois
for r in rois:
    mean_zscores = mean_zscore_roi(one_rec, r)
    z_vel_30 = [z for z, a in zip(mean_zscores, angul_vel) if a == 30.0]
    z_vel_minus30 = [z for z, a in zip(mean_zscores, angul_vel) if a == -30.0]
    z_vel_0 = [z for z, a in zip(mean_zscores, angul_vel) if a == 0.0]

    fig, ax = plt.subplots()
    ax.boxplot(z_vel_0, positions=[1])
    ax.boxplot(z_vel_minus30, positions=[2])
    ax.boxplot(z_vel_30, positions=[3])
    ax.set_xticklabels(['0vel', '-30vel', '30vel'])
    plt.show()
""""

"""
plt.plot(time, one_rec[roi].zscore)
plt.scatter(np.sort(start_time), np.ones_like(start_time), c='r')
plt.scatter(np.sort(end_time), np.ones_like(end_time))
plt.scatter((start_time) + (end_time - start_time)/2, mean_zscore, c='k')
plt.show()

for i, roi in enumerate(one_rec[:20]):
    dff = roi.dff
    plt.plot(i + (dff - dff.min()) / (dff.max() - dff.min()),
             color='black', linewidth='1.')
plt.show()
"""

"""
roi = 10
z_vel_30 = [z for z,a in zip(mean_zscore, angul_vel) if a == 30.0]
z_vel_minus30 = [z for z,a in zip(mean_zscore, angul_vel) if a == -30.0]
z_vel_0 = [z for z, a in zip(mean_zscore, angul_vel) if a == 0.0]
"""

"""
correlate repeats with each other:
take first two repeats of the same phase series to find cells that respond over time
use dff
"""