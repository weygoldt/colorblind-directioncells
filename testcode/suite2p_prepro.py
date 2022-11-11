import h5py as h5
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy.stats import spearmanr, pearsonr
from tqdm import tqdm
from vxtools.summarize.structure import SummaryFile
from itertools import combinations
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
    
def mean_dff_roi(one_recording, roi):
    """Calculates the mean dff between the start and end point of the stimulus of one single ROI

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

    mean_dff = []
    for snip in time_snippets:
        zscores_snip = one_recording[roi].dff[snip]
        mean_dff.append(np.mean(zscores_snip))

    return mean_dff

def repeats(angul_vel, nrepeats=3):
    """Calculate the repeats of one Recroding with the angular velocity 

    Parameters
    ----------
    angul_vel : list or ndarray
        all the angular velocity in one recording including nan 
    nrepeats : int, optional
        how many repeats there are in the protocol , by default 3

    Returns
    -------
    tuple 
        Index tuple where one can index the recoring with nrepeats 

    Raises
    ------
    ValueError
        if the recording is not dividable by nrepeats 
    """
    nonnan_angul_vel = [x for x in angul_vel if np.isnan(x) == False]

    nr = len(nonnan_angul_vel)/nrepeats
    if nr % 1 != 0:
        raise ValueError(f'Cant divide by {nrepeats}!')
    inx  = []
    for i in np.arange(1, nrepeats+1):
        if i == 1:
            inx.append((0,nr))
        else:
            inx.append(((i-1)*nr,nr*i))
    inx = np.array(inx, dtype=int)
    return inx

def corr_repeats(mean_score, inx):
    """Calculate the the correlation of one ROI within nrepeats

    Parameters
    ----------
    mean_score : list or 1darray
        mean_score values of 
    inx : _type_
        _description_

    Returns
    -------
    _type_
        _description_
    """
    z = []
    for i in inx:
        z.append(mean_score[i[0]:i[-1]])
    combs = [comb for comb in combinations(z, 2)]
    spear = []
    for co in combs:
        spear.append(spearmanr(co[0], co[1])[0])
    
    spear_mean = np.mean(spear)
    
    return spear_mean

def active_rois(one_recording, inx, threshold=0.6 ):
    spearmeans = []
    for r in tqdm(range(len(one_recording))):
        mean_zscore= mean_dff_roi(one_recording, r)
        spear_mean = corr_repeats(mean_zscore, inx)
        spearmeans.append(spear_mean)
    index = np.arange(len(spearmeans), dtype=int)
    active_roi = index[np.array(spearmeans) >= threshold]
    spearmeans_a_roi = np.array(spearmeans)[active_roi] 
    result = np.empty((len(active_roi),2))
    active_spear = np.vstack((active_roi, spearmeans_a_roi)).T
    result[:,0] = active_roi
    result[:,1] = spearmeans_a_roi
    active_spear_sorted = sorted(result, key=lambda x : x[1], reverse=True)
    return active_spear_sorted

def plot_ang_velocity(onerecording, rois):
    for r in rois:
        mean_zscores = mean_dff_roi(onerecording, r)
        z_vel_30 = [z for z, a in zip(mean_zscores, angul_vel) if a == 30.0]
        z_vel_minus30 = [z for z, a in zip(mean_zscores, angul_vel) if a == -30.0]
        z_vel_0 = [z for z, a in zip(mean_zscores, angul_vel) if a == 0.0]

        fig, ax = plt.subplots()
        ax.boxplot(z_vel_0, positions=[1])
        ax.boxplot(z_vel_minus30, positions=[2])
        ax.boxplot(z_vel_30, positions=[3])
        ax.set_xticklabels(['0vel', '-30vel', '30vel'])
        plt.show()

embed()
exit()
one_rec = data_one_rec_id(f, 5)
time = one_rec[0].times
start_time, end_time, angul_vel, angul_pre, rgb_1, rgb_2 = get_attributes(
    one_rec)
inx = repeats(angul_vel)
active_spear = active_rois(one_rec,inx, threshold=0.3)

active_dff = [one_rec[ar].dff for ar in active_spear]

dff = np.array([roi.dff for roi in one_rec])
o
plt.imshow(dff)
plt.show()


dff = np.array([roi.dff for roi in one_rec])

for i, roi in enumerate(f.rois):
    dff = roi.dff
    plt.plot(i + (dff - dff.min()) / (dff.max() - dff.min()),
             color='black', linewidth='1.')
plt.show()



embed()
exit()

"""
rposs = []
rnegs = []
for roi in tqdm(range(len(one_rec))):

    # make logical arrays for left and right turning motion
    mean_zscore = mean_zscore_roi(one_rec, roi)[1:-1]
    angveloc_pos = [1 if a == 30.0 else 0 for a in angul_vel][1:-1]
    angveloc_neg = [1 if a == -30.0 else 0 for a in angul_vel][1:-1]

    # compute correlation
    rposs.append(spearmanr(mean_zscore, angveloc_pos))
    rnegs.append(spearmanr(mean_zscore, angveloc_neg))


for i, s in enumerate(rposs):
    plt.scatter(i, s[0])

"""
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
"""

"""
plt.plot(time, one_rec[roi].zscore)
plt.scatter(np.sort(start_time), np.ones_like(start_time), c='r')
plt.scatter(np.sort(end_time), np.ones_like(end_time))
plt.scatter((start_time) + (end_time - start_time)/2, mean_zscore, c='k')
plt.show()


"""

"""
roi = 10
z_vel_30 = [z for z, a in zip(mean_zscore, angul_vel) if a == 30.0]
z_vel_minus30 = [z for z, a in zip(mean_zscore, angul_vel) if a == -30.0]
z_vel_0 = [z for z, a in zip(mean_zscore, angul_vel) if a == 0.0]
"""

"""
correlate repeats with each other:
take first two repeats of the same phase series to find cells that respond over time
use dff
"""