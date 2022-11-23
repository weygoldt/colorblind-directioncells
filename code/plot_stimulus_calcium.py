
from copy import deepcopy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import plottools.colors as c
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
mf.filter_rois(indices_thresh)

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

mfclock = deepcopy(mf)
mfcclock = deepcopy(mf)
mfclock.filter_rois(index_clock)
mfcclock.filter_rois(index_cclock)

# build the stimulus
colors = c.colors_vivid
red = c.lighter(colors['red'], 0.8)
green = c.lighter(colors['green'], 0.8)

reds = np.linspace(0, 1, 6)
greens = np.linspace(0, 1, 6)

r_rgb = [c.darker(red, x) for x in reds][::-1]
g_rgb = [c.darker(green, x) for x in greens]

rgb_matrix = []
for i1 in range(6):

    r = r_rgb[i1]

    for i2 in range(6):

        g = g_rgb[i2]

        rg_rgb = [g, r]

        rgb_matrix.append(rg_rgb)

# dim = np.shape(rgb_matrix)
# rgb_matrix_flat = rgb_matrix.reshape(dim[0]*dim[1], dim[2], dim[3])
fill_r = [[0, 2], [1, 3]]
fill_g = [[1, 3], [2, 4]]

# gridspec inside gridspec
fig = plt.figure(figsize=(22*ps.cm, 10*ps.cm))
gs0 = gridspec.GridSpec(1, 4, figure=fig, width_ratios=[1, 0.2, 1, 0.05])

# axes for stimulus
gs00 = gs0[0].subgridspec(6, 6)
stim_axs = [fig.add_subplot(i) for i in gs00]

# axis for heatmap
gs01 = gs0[2].subgridspec(1, 1)
heatm_ax = fig.add_subplot(gs01[0])

gs02 = gs0[3].subgridspec(1, 1)
colorbar = fig.add_subplot(gs02[0])

# plot the stimulus
for axis, rgbs in zip(np.array(stim_axs).flat, rgb_matrix):
    for i in range(len(fill_r)):
        axis.axvspan(fill_r[0][i], fill_r[1][i], color=rgbs[0])
        axis.axvspan(fill_g[0][i], fill_g[1][i], color=rgbs[1])
        axis.axis('off')

# plot the heatmap
# >>> DISCLAIMER this cell can probably be made much nicer but its half past 10 on a saturday


def unique_combinations2d(list1, list2):
    """Combine elments of two lists uniquely."""
    return [(x, y) for x in list1 for y in list2]


reds = np.unique(mf.red)
greens = np.unique(mf.green)
combs = unique_combinations2d(reds, greens)
index = np.arange(len(mf.red))
matrix = np.full((6, 6), np.nan)

signal1 = []
signal2 = []
for comb in combs:

    loc = index[(mf.red == comb[0]) & (mf.green == comb[1])]

    signal1.append(mfclock.zscores[:, loc])
    signal2.append(mfcclock.zscores[:, loc])

signal1 = np.mean(np.mean(signal1, axis=1), axis=1)
signal2 = np.mean(np.mean(signal2, axis=1), axis=1)
signal = np.mean(np.array([signal1, signal2]), axis=0)


# put into matrix
for i, s in enumerate(signal):
    idx = np.unravel_index(i, np.shape(matrix))
    matrix[idx] = s

hm = heatm_ax.imshow(
    matrix,
    interpolation="none",
    vmin=-0.2,
    vmax=0.2,
    aspect='equal',
    extent=[-0.1, 1.1, -0.1, 1.1],
    origin='lower'
)

heatm_ax.set_xticks(np.linspace(0, 1, 6))
heatm_ax.set_yticks(np.linspace(0, 1, 6))

heatm_ax.set_xticklabels(list(map(str, np.round(np.unique(mf.red), 2))))
heatm_ax.set_yticklabels(list(map(str, np.round(np.unique(mf.red), 2))))

heatm_ax.set_xlabel('green contr.')
heatm_ax.set_ylabel('red contr.')
# heatm_ax.set_xticks(np.linspace(0, 1, 6))
# heatm_ax.set_yticks(np.linspace(0, 1, 6))

# xaxis = np.linspace(-0.1, 1.1, 6)
# yaxis = np.linspace(-0.1, 1.1, 6)
# xvals = xaxis[np.where(matrix == np.min(matrix, axis=0))[0]]
# yvals = xaxis[np.where(matrix == np.min(matrix, axis=0))[1]]
# heatm_ax.plot(xvals, yvals, c='white')

cbar = fig.colorbar(hm, cax=colorbar, shrink=0.7, pad=0)
cbar.ax.set_ylabel('mean z-score')

plt.subplots_adjust(left=0.015, right=0.885, top=0.950,
                    bottom=0.170, hspace=0.250, wspace=0.1)
fs.doublesave('../poster/figs/calcium_stim')
plt.show()
