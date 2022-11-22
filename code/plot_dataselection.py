from copy import deepcopy

import cmocean.cm as cm
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plottools.colors as c
import scipy.stats
from scipy.stats import pearsonr
from sklearn.metrics import auc
from vxtools.summarize.structure import SummaryFile

import modules.functions as fs
from modules.contrast import (phase_activity, rg_activity_trash,
                              selective_rois_trash)
from modules.dataloader import MultiFish, SingleFish
from modules.plotstyle import PlotStyle

ps = PlotStyle()

# now load the data
data1 = '../data/data1/'
data2 = '../data/data2/'
data3 = '../data/data3/'

f1 = SummaryFile(data1 + 'Summary.hdf5')
f2 = SummaryFile(data2 + 'Summary.hdf5')
f3 = SummaryFile(data3 + 'Summary.hdf5')

good_recs1 = np.arange(3, 15)
good_recs2 = [0, 1, 2, 4, 5]
good_recs3 = [0, 1, 2, 3, 4]

# load matrix of all rois of all layers with good rois
d1 = SingleFish(f1, good_recs1, overwrite=False, behav=False)
d2 = SingleFish(f2, good_recs2, overwrite=False, behav=True)
d3 = SingleFish(f3, good_recs3, overwrite=False, behav=True)

# load all fish in multifish class
mf = MultiFish([
    d1,
    d2,
    d3
])

mf_all = MultiFish([
    d1,
    d2,
    d3
])


target_auc = 0.2  # probability threshold

# compute the correlations and indices sorted by correlations
mf_thresh = deepcopy(mf)
mf_thresh.phase_means()
indices, corrs = mf_thresh.responding_rois(mf_thresh.dffs, nrepeats=3)

# make a histpgram
counts, edges = np.histogram(corrs, bins=50, range=(-1, 1), density=True)

# compute a gaussian KDE
xkde, kde = fs.kde1d(corrs, 0.02, xlims=[edges.min(), edges.max()])

# create empty arrays for the gradient and index
gradient = np.zeros_like(xkde[:-1])
index = np.arange(len(xkde[:-1]))

# compute the gradient between the target and actual auc
for i in range(len(xkde[:-1])):
    area = auc(xkde[i:], kde[i:])
    gradient[i] = abs(area-target_auc)

# find the index where the gradient is smallest
idx = index[gradient == gradient.min()][0]

# get the threshold for correlation coefficients here
thresh = xkde[idx]
print(f"{thresh=}")

# only take the indices where correlation coefficients crossed the thresh
indices_thresh = indices[corrs > thresh]
corrs_thresh = corrs[corrs > thresh]

# create the subset of the dataset for these indices
mf_thresh.filter_rois(indices_thresh)
mf.filter_rois(indices_thresh)
mf_all.filter_rois(indices_thresh)

# compute the mean in each phase
mf.repeat_means(3)
mf_all.repeat_means(3)

# build motion regressor and correlate
motion = np.array([1 if x != 0 else 0 for x in mf_thresh.ang_velocs])
corr_motion = np.array([pearsonr(x, motion)[0] for x in mf_thresh.dffs])

# clockwise motion regressor
clock = np.array([1 if x > 0 else 0 for x in mf_thresh.ang_velocs])
corr_clock = np.array([pearsonr(x, clock)[0] for x in mf_thresh.dffs])

# counterclockwise motion regressor
cclock = np.array([1 if x < 0 else 0 for x in mf_thresh.ang_velocs])
corr_cclock = np.array([pearsonr(x, cclock)[0] for x in mf_thresh.dffs])

# correlation coefficient threshold
thresh = 0.3

# get index of active ROIs for correlation threshold for clockwise and
# counterclockwise regressor correlations
index_clock = np.arange(len(mf.dffs[:, 0]))[corr_clock > thresh]
index_cclock = np.arange(len(mf.dffs[:, 0]))[corr_cclock > thresh]

# create copies of dataset
mfclock = deepcopy(mf)
mfcclock = deepcopy(mf)

# filter direction selective rois
mfclock.filter_rois(index_clock)
mfcclock.filter_rois(index_cclock)

# remove these from the active roi dataset to also plot some non responding
remove_rois = np.asarray(fs.flatten([mfclock.rois, mfcclock.rois]))
remove_recs = np.asarray(fs.flatten([mfclock.recs, mfcclock.recs]))
remove_pairs = [[x, y] for x, y in zip(remove_rois, remove_recs)]
true_pairs = [[x, y] for x, y in zip(mf.rois, mf.recs)]
indices = np.arange(len(mf.rois))
remove_indices = np.asarray(
    [i for i, tp in zip(indices, true_pairs) if tp not in remove_pairs])
mf.filter_rois(remove_indices)

# remove from mf_all what is already included in mf,mfclock and mfcclock
remove_rois = np.asarray(fs.flatten([mfclock.rois, mfcclock.rois, mf.rois]))
remove_recs = np.asarray(fs.flatten([mfclock.rois, mfcclock.rois, mf.rois]))
remove_pairs = [[x, y] for x, y in zip(remove_rois, remove_recs)]
true_pairs = [[x, y] for x, y in zip(mf_all.rois, mf_all.recs)]
indices = np.arange(len(mf_all.rois))
remove_indices = np.asarray(
    [i for i, tp in zip(indices, true_pairs) if tp not in remove_pairs])
mf_all.filter_rois(remove_indices)


# remove pauses
pause_index = np.arange(len(mf.ang_velocs))[mf.ang_velocs != 0]
mf.filter_phases(pause_index)
mfclock.filter_phases(pause_index)
mfcclock.filter_phases(pause_index)
mf_all.filter_phases(pause_index)

# make sorter
sorter_indices = np.arange(len(mf.red))
sorter = [(x, y, z, i)
          for x, y, z, i in zip(mf.ang_velocs, mf.red, mf.green, sorter_indices)]

sorted_sorter = sorted(sorter)
index = [x[3] for x in sorted_sorter]

mf.filter_phases(index)
mfclock.filter_phases(index)
mfcclock.filter_phases(index)
mf_all.filter_phases(index)

dffs1 = np.asarray([np.ravel(x) for x in mfcclock.dffs])
dffs2 = np.asarray([np.ravel(x) for x in mfclock.dffs])
dffs3 = np.asarray([np.ravel(x) for x in mf.dffs[:75, :]])
dffs4 = np.asarray([np.ravel(x) for x in mf_all.dffs[-75:, :]])

dffs = np.concatenate([dffs1, dffs2, dffs3, dffs4])


newtime = np.arange(0, 0.1*len(dffs[0, :]), 0.1)
new_start_times = np.arange(0, newtime[-1], 4)
new_stop_times = np.arange(3.9, newtime[-1], 4)

colors = c.colors_vivid
red = c.lighter(colors['red'], 0.8)
green = c.lighter(colors['green'], 0.8)
new_reds = np.ravel([np.ones(6) * x for x in np.linspace(0, 1, 6)])
new_reds = np.ravel([new_reds, new_reds])
new_greens = np.ravel([np.linspace(0, 1, 6) for i in range(6)])
new_greens = np.ravel([new_greens, new_greens])

extent = (np.min(newtime), np.max(newtime), 0, len(dffs[:, 0]))

fig, ax = plt.subplots(2, 1, figsize=(30*ps.cm, 20*ps.cm),
                       height_ratios=[0.05, 1], sharex=True)


ax[1].imshow(dffs, aspect='auto', extent=extent)

for i, (r, g) in enumerate(zip(new_reds, new_greens)):
    start = new_start_times[i]
    end = new_stop_times[i]

    rc = c.darker(red, r)
    gc = c.darker(green, g)

    ax[0].axvspan(start, start + (end-start)/2, color=rc)
    ax[0].axvspan(start + (end-start)/2, end, color=gc)

ax[0].axis('off')

ax[0].axvline(np.max(newtime)/2, color='white', lw=1.5,)
ax[1].axvline(np.max(newtime)/2, color='white', lw=1, ls='dashed')
ax[1].axhline(75, color='white', lw=1, ls='dashed')
ax[1].axhline(150, color='white', lw=1, ls='dashed')
ax[1].axhline(224, color='white', lw=1, ls='dashed')

ax[1].set_ylabel('ROI')
ax[1].set_xlabel('time [s]')

plt.figtext(0.96, (0.72), 'Countercl.', rotation=90)
plt.figtext(0.96, (0.55), 'Clockw.', rotation=90)
plt.figtext(0.96, (0.32), 'Responding', rotation=90)
plt.figtext(0.96, (0.085), 'Non-responding', rotation=90)

plt.figtext(0.6, (0.9), 'Countercl. stim.', rotation=0)
plt.figtext(0.8, (0.9), 'Clockw. stim.', rotation=0)

plt.subplots_adjust(left=0.065, right=0.940, top=0.900,
                    bottom=0.080, hspace=0.040, wspace=0.000)
fs.doublesave('../plots/testimg')
plt.show()
