
from copy import deepcopy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from IPython import embed
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
d2 = SingleFish(f2, good_recs2, overwrite=False,)
d1 = SingleFish(f1, good_recs1, overwrite=False,)
d3 = SingleFish(f3, good_recs3, overwrite=False,)

# load all fish in multifish class
mf = MultiFish([
    d1,
    d2,
    d3
])

mfcopy = deepcopy(mf)
mfcopy1 = deepcopy(mf)

mf.phase_means()
target_auc = 0.2  # probability threshold

# compute the correlations and indices sorted by correlations
indices, corrs = mf.responding_rois(mf.dffs, nrepeats=3)

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

mfcopy.filter_rois(indices_thresh)

fig, ax = plt.subplots(figsize=(ps.cm*21, ps.cm*12))
temp_zscores = np.asarray([fs.flatten(x) for x in mfcopy.zscores])
temp_zscores1 = np.asarray([fs.flatten(x) for x in mfcopy1.zscores])
x = np.concatenate((temp_zscores[10:15], temp_zscores1[10:15]))

n = 10
min = []
max = []
l = [1, 2, 3, 5, 9]
for i, roi in enumerate(l):
    dff = x[roi, :]
    max.append(np.max(dff))
    min.append(np.min(dff))

for i, roi in enumerate(l):
    dff = x[roi, :]
    ax.plot(mfcopy.times/60, i + (dff - np.min(min)) /
            (np.max(max) - np.min(min)), color='black', linewidth='1.')


ax.set_xlabel("Time [min]", fontsize=15)
ax.set_ylabel("ROIs $\\frac{\Delta F}{F}$ ",     fontsize=15, rotation=90,)
ax.set_yticks([])
ax.set_ylim(0, 6)
ax.set_xticks(np.round(np.arange((mfcopy.times[0])/60, 30.1, 5), 1))
ax.vlines(9.631, -1, 6, ls='dashed', color='r')
ax.vlines(19.23, -1, 6, ls='dashed', color='r')


ax.set_xticklabels(
    np.round(np.arange((mfcopy.times[0])/60, 30.1, 5), 1), fontsize=13)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)
ax.spines["left"].set_visible(False)

fs.doublesave('../poster/figs/autocorrelation')


# clockwise motion regressor
clock = np.array([1 if x > 0 else 0 for x in mf.ang_velocs])
corr_clock = np.array([pearsonr(x, clock)[0] for x in mf.dffs])

# counterclockwise motion regressor
cclock = np.array([1 if x < 0 else 0 for x in mf.ang_velocs])
corr_cclock = np.array([pearsonr(x, cclock)[0] for x in mf.dffs])

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


temp_clock = mfcclock.zscores[:5]

clock_acr = np.array(clock)
cclock_acr = np.array(cclock)

# correlate
mf.filter_rois(indices_thresh)
corr_clock_acr = np.array([pearsonr(clock_acr, x)[0] for x in mf.zscores])
corr_cclock_acr = np.array([pearsonr(cclock_acr, x)[0] for x in mf.zscores])

# make index vector
index = np.arange(len(corr_clock_acr))

# make threshold
thresh = 0.3

# use threshold to get index
index_clock_acr = index[corr_clock_acr > thresh]
index_cclock_acr = index[corr_cclock_acr > thresh]

# use the index to get the dff data
dffs_clock_acr = mf.zscores[index_clock_acr, :][:5]
dffs_cclock_acr = mf.zscores[index_cclock_acr, :][:5]


# plot them
idx = np.arange(0, 121, 1)
fig, ax = plt.subplots(2, 1, figsize=(20*ps.cm, 12*ps.cm),
                       constrained_layout=True, sharex=True, sharey=True)

ax[0].set_title('Clockwise', loc='left')
ax[0].plot(mf.times[idx], clock_acr[idx], c='k',
           label='stimulus', zorder=100, lw=2)

for i in range(len(dffs_clock_acr[:, 0])):
    clock_acr_dff = dffs_clock_acr[i]
    ax[0].plot(mf.times[idx], clock_acr_dff[idx] -
               clock_acr_dff[idx].min(), label='dff', alpha=1, color='darkgray')

ax[1].set_title('Counterclockwise', loc='left')
ax[1].plot(mf.times[idx], cclock_acr[idx], c='k',
           label='stimulus', zorder=100, lw=2)

for i in range(len(dffs_cclock_acr[:, 0])):
    cclock_acr_dff = dffs_cclock_acr[i]
    ax[1].plot(mf.times[idx], cclock_acr_dff[idx] -
               cclock_acr_dff[idx].min(), label='dff', alpha=1, color='darkgray')

ax[1].set_xticks(np.arange(0, mf.times[idx[-1]], 50))
[x.spines["right"].set_visible(False) for x in ax]
[x.spines["top"].set_visible(False) for x in ax]
fig.supxlabel('Time [s]')
fig.supylabel('Zscores')
# plot them
idx = np.arange(0, 121, 1)
fig, ax = plt.subplots(2, 1, figsize=(20*ps.cm, 12*ps.cm),
                       constrained_layout=True, sharex=True, sharey=True)

ax[0].set_title('Clockwise', loc='left')
ax[0].plot(mf.times[idx], clock_acr[idx], c='k',
           label='stimulus', zorder=100, lw=2)

for i in range(len(dffs_clock_acr[:, 0])):
    clock_acr_dff = dffs_clock_acr[i]
    ax[0].plot(mf.times[idx], clock_acr_dff[idx] -
               clock_acr_dff[idx].min(), label='dff', alpha=1, color='darkgray')

ax[1].set_title('Counterclockwise', loc='left')
ax[1].plot(mf.times[idx], cclock_acr[idx], c='k',
           label='stimulus', zorder=100, lw=2)

for i in range(len(dffs_cclock_acr[:, 0])):
    cclock_acr_dff = dffs_cclock_acr[i]
    ax[1].plot(mf.times[idx], cclock_acr_dff[idx] -
               cclock_acr_dff[idx].min(), label='dff', alpha=1, color='darkgray')

ax[1].set_xticks(np.arange(0, mf.times[idx[-1]], 50))
[x.spines["right"].set_visible(False) for x in ax]
[x.spines["top"].set_visible(False) for x in ax]

fig.supxlabel('Time [s]', fontsize=15)
fig.supylabel('Zscores' , fontsize=15)
fs.doublesave('../poster/figs/regressor')
plt.show()
