from copy import deepcopy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from IPython import embed
from scipy.stats import pearsonr
from sklearn.metrics import auc
from tqdm import tqdm
from vxtools.summarize.structure import SummaryFile

import modules.functions as fs
from modules.contrast import (phase_activity, rg_activity_trash,
                              selective_rois_trash)
from modules.dataloader import MultiFish, SingleFish
from modules.plotstyle import PlotStyle


class rg_activity:
    def __init__(self, selective_rois, contr1, contr2):

        self.__rois = selective_rois
        self.__contr1 = contr1
        self.__contr2 = contr2
        self.__index = np.arange(len(self.__contr1))
        self.contr1 = np.unique(self.__contr1[~np.isnan(self.__contr1)])
        self.contr2 = np.unique(self.__contr2[~np.isnan(self.__contr2)])
        self.zscores = []
        # self.mean_dffs = []
        self.contr1_index = []
        self.contr2_index = []
        self.rois = self.__rois.rois
        self.recs = self.__rois.recs

        for c1 in self.contr1:

            self.contr1_index.append(c1)
            idx = self.__index[self.__contr1 == c1]
            cat_zscores = self.__rois.zscores[:, idx]
            # mean_dffs = np.mean(cat_dffs, axis=1)
            self.contr2_index.append(self.__contr2[idx])
            self.zscores.append(cat_zscores)
            # self.mean_dffs.append(mean_dffs)

        # self.mean_dffs = np.array(self.mean_dffs)
        self.contr1_index = np.array(self.contr1_index)
        self.contr2_index = np.array(self.contr2_index)


class rg_activity_mating:
    def __init__(self, rg_activities):

        self.zscores = np.concatenate(
            [x.zscores for x in rg_activities], axis=1)
        self.contr1_index = [x.contr1_index for x in rg_activities][0]
        self.contr2_index = [x.contr2_index for x in rg_activities][0]
        self.contr1 = [x.contr1 for x in rg_activities][0]
        self.contr2 = [x.contr2 for x in rg_activities][0]


def dataconverter(data, nresamples=10000, subsetsize_resampling=0.95):

    contrast1 = []
    contrast2 = []
    mean_zscores = []
    q025_zscores = []
    q975_zscores = []

    # iterate through contrast levels
    for it1, c1 in tqdm(enumerate(data.contr1)):

        # collect zscore and contrast data here
        zscores = []
        contrasts1 = []
        contrasts2 = []

        # for each contrast1 level, go through all contrast2 levels
        for zsc in np.array(data.zscores)[data.contr1 == c1]:

            # go through each single ROI and append contrast levels and zscores
            for roi in range(len(zsc[:, 0])):

                # get the contrast2 for this specific contrast1
                contr2 = np.array(data.contr2_index)[data.contr1 == c1][0]

                # sort by contrast level
                sort_zsc = zsc[roi, np.argsort(contr2)]
                sort_contr2 = contr2[np.argsort(contr2)]

                # append contrasts and zscore matrices
                contrasts1.append(np.ones_like(sort_contr2)*c1)
                contrasts2.append(sort_contr2)
                zscores.append(sort_zsc)

        # for each contrast 2 category go through and compute the means and std
        contrasts1 = np.asarray(np.ravel(contrasts1))
        contrasts2 = np.asarray(np.ravel(contrasts2))
        zscores = np.asarray(np.ravel(zscores))
        contr2_lvls = np.unique(contrasts2)

        mean_zsc = []
        q025_zsc = []
        q975_zsc = []

        for c2 in contr2_lvls:

            # get zscores for this condition
            zscs = zscores[(contrasts2 == c2) & (contrasts1 == c1)]

            # resample them
            subsetsize = int(np.round(len(zscs)*subsetsize_resampling))

            resamples = []
            for i in range(nresamples):
                subset = np.random.choice(zscs, size=subsetsize, replace=False)
                resamples.append(subset)

            resamples = np.array(np.ravel(resamples))

            mean_zsc.append(np.mean(resamples))
            q025_zsc.append(np.percentile(resamples, 2.5))
            q975_zsc.append(np.percentile(resamples, 97.5))

        mean_zsc = np.asarray(mean_zsc)
        q025_zsc = np.asarray(q025_zsc)
        q975_zsc = np.asarray(q975_zsc)

        mean_zscores.append(mean_zsc)
        q025_zscores.append(q025_zsc)
        q975_zscores.append(q975_zsc)

        contrast1.append(np.ones_like(contr2_lvls)*c1)
        contrast2.append(contr2_lvls)

    contrast1 = np.asarray(contrast1)
    contrast2 = np.asarray(contrast2)
    mean_zscores = np.asarray(mean_zscores)
    q025_zscores = np.asarray(q025_zscores)
    q975_zscores = np.asarray(q975_zscores)

    return contrast1, contrast2, mean_zscores, q025_zscores, q975_zscores


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
d1 = SingleFish(f1, good_recs1, overwrite=False)
d2 = SingleFish(f2, good_recs2, overwrite=False)
d3 = SingleFish(f3, good_recs3, overwrite=False)

# load all fish in multifish class
mf = MultiFish([
    d1,
    # d2,
    # d3
])

# compute the mean in each phase
mf.phase_means()

# threshold the active ROIs

target_auc = 0.05  # probability threshold

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

# only take the indices where correlation coefficients crossed the thresh
indices_thresh = indices[corrs > thresh]
corrs_thresh = corrs[corrs > thresh]

# create the subset of the dataset for these indices
mf.filter_rois(indices_thresh)

print(f"Old number of indices: {len(indices)}")
print(f"New number of indices: {len(indices_thresh)}")
print(f"Number of ROIs after thresholding: {len(mf.dffs[:,0])}")

# build motion regressor and correlate
motion = np.array([1 if x != 0 else 0 for x in mf.ang_velocs])
corr_motion = np.array([pearsonr(x, motion)[0] for x in mf.dffs])

# clockwise motion regressor
clock = np.array([1 if x > 0 else 0 for x in mf.ang_velocs])
corr_clock = np.array([pearsonr(x, clock)[0] for x in mf.dffs])

# counterclockwise motion regressor
cclock = np.array([1 if x < 0 else 0 for x in mf.ang_velocs])
corr_cclock = np.array([pearsonr(x, cclock)[0] for x in mf.dffs])

# correlation coefficient threshold
thresh = 0.4

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

# modify motion stim to remove all phases without motion
clock = np.asarray([np.nan if x == 0 else x for x in clock])
cclock = np.asarray([np.nan if x == 0 else x for x in cclock])

# make regressors
red_clock = clock * mf.red
green_clock = clock * mf.green
red_cclock = cclock * mf.red
green_cclock = cclock * mf.green

# extract activity at phases
rg_clock_data = rg_activity(mfclock, red_clock, green_clock)
rg_cclock_data = rg_activity(mfcclock, red_cclock, green_cclock)
gr_clock_data = rg_activity(mfclock, green_clock, red_clock)
gr_cclock_data = rg_activity(mfcclock, green_cclock, red_cclock)

# make atasets for resampling
bootstr_data_clock = deepcopy(mfclock)
bootstr_data_cclock = deepcopy(mfcclock)

# shuffle zscores
for i in range(len(bootstr_data_clock.zscores[:, 0])):
    clock_index = np.arange(len(mf.ang_velocs))[mf.ang_velocs > 0]
    clock_data = bootstr_data_clock.zscores[i, clock_index]
    np.random.shuffle(clock_data)
    bootstr_data_clock.zscores[i, clock_index] = clock_data

for i in range(len(bootstr_data_cclock.zscores[:, 0])):
    cclock_index = np.arange(len(mf.ang_velocs))[mf.ang_velocs < 0]
    cclock_data = bootstr_data_cclock.zscores[i, cclock_index]
    np.random.shuffle(cclock_data)
    bootstr_data_cclock.zscores[i, cclock_index] = cclock_data

rg_bootstr_clock = rg_activity(bootstr_data_clock, red_clock, green_clock)
rg_bootstr_cclock = rg_activity(bootstr_data_cclock, red_cclock, green_cclock)
gr_bootstr_clock = rg_activity(bootstr_data_clock, green_clock, red_clock)
gr_bootstr_cclock = rg_activity(bootstr_data_cclock, green_cclock, red_cclock)

rg_bootstr = rg_activity_mating([rg_bootstr_cclock, rg_bootstr_clock])
gr_bootstr = rg_activity_mating([gr_bootstr_cclock, gr_bootstr_clock])

fig = plt.figure(figsize=(30*ps.cm, 20*ps.cm))

# build the subfigures
(subfig_l, subfig_r) = fig.subfigures(1, 2, hspace=0.05, width_ratios=[1, 1])

gs0 = gridspec.GridSpec(2, 3, figure=subfig_l)
stim_axs1 = [subfig_l.add_subplot(i) for i in gs0]

gs1 = gridspec.GridSpec(2, 3, figure=subfig_r)
stim_axs2 = [subfig_r.add_subplot(i) for i in gs1]

# make axes shared
stim_axs1[0].get_shared_x_axes().join(stim_axs1[0], *stim_axs1)
stim_axs1[0].get_shared_y_axes().join(stim_axs1[0], *stim_axs1)
stim_axs2[0].get_shared_x_axes().join(stim_axs2[0], *stim_axs2)
stim_axs2[0].get_shared_y_axes().join(stim_axs2[0], *stim_axs2)

# plot the left side of the plot
colors = [ps.orange, ps.red, ps.blue]
labels = ['clockw.-selective', 'countercl.-selective', 'bootstrap']

# get the data
contrast1, contrast2, mean_zscores_c, q025_zscores_c, q975_zscores_c = dataconverter(
    rg_clock_data)
_, _, mean_zscores_cc, q025_zscores_cc, q975_zscores_cc = dataconverter(
    rg_cclock_data)
_, _, mean_zscores_brg, q025_zscores_brg, q975_zscores_brg = dataconverter(
    rg_bootstr)

for i, ax in enumerate(stim_axs1):

    ax.plot(contrast2[i], mean_zscores_c[i], lw=2, c=ps.red)
    ax.plot(contrast2[i], mean_zscores_cc[i], lw=2, c=ps.orange)
    ax.plot(contrast2[i], mean_zscores_brg[i], lw=2, c=ps.blue)

    ax.fill_between(contrast2[i], q025_zscores_c[i],
                    q975_zscores_c[i], alpha=0.2, color=ps.red)
    ax.fill_between(contrast2[i], q025_zscores_cc[i],
                    q975_zscores_cc[i], alpha=0.2, color=ps.orange)
    ax.fill_between(contrast2[i], q025_zscores_brg[i],
                    q975_zscores_brg[i], alpha=0.2, color=ps.blue)

    ax.axhline(0, lw=1, ls='dashed', c='darkgray')

# plot the right side of the plot
colors = [ps.orange, ps.red, ps.blue]
labels = ['clockw.-selective', 'countercl.-selective', 'bootstrap']

# get the data
contrast1, contrast2, mean_zscores_c, q025_zscores_c, q975_zscores_c = dataconverter(
    gr_clock_data)
_, _, mean_zscores_cc, q025_zscores_cc, q975_zscores_cc = dataconverter(
    gr_cclock_data)
_, _, mean_zscores_bgr, q025_zscores_bgr, q975_zscores_bgr = dataconverter(
    rg_bootstr)

for i, ax in enumerate(stim_axs2):

    ax.plot(contrast2[i], mean_zscores_c[i])
    ax.plot(contrast2[i], mean_zscores_cc[i])
    ax.plot(contrast2[i], mean_zscores_bgr[i])

    ax.fill_between(contrast2[i], q025_zscores_c[i],
                    q975_zscores_c[i], alpha=0.2)
    ax.fill_between(contrast2[i], q025_zscores_cc[i],
                    q975_zscores_cc[i], alpha=0.2)
    ax.fill_between(contrast2[i], q025_zscores_bgr[i],
                    q975_zscores_bgr[i], alpha=0.2)

    ax.axhline(0, lw=1, ls='dashed', c='darkgray')

# remove axes on right and top of plots
[x.spines["right"].set_visible(False) for x in np.asarray(stim_axs1)]
[x.spines["top"].set_visible(False) for x in np.asarray(stim_axs1)]
[x.spines["right"].set_visible(False) for x in np.asarray(stim_axs2)]
[x.spines["top"].set_visible(False) for x in np.asarray(stim_axs2)]

# turn y axes off where not needed
yaxes_off = [1, 2, 4, 5]
[x.get_yaxis().set_visible(False) for x in np.asarray(stim_axs1)[yaxes_off]]
[x.get_yaxis().set_visible(False) for x in np.asarray(stim_axs2)[yaxes_off]]
[x.spines["left"].set_visible(False) for x in np.asarray(stim_axs1)[yaxes_off]]
[x.spines["left"].set_visible(False) for x in np.asarray(stim_axs2)[yaxes_off]]

# turn x axes off where they are not needed
xaxes_off = [0, 1, 2]
[x.get_xaxis().set_visible(False) for x in np.asarray(stim_axs1)[xaxes_off]]
[x.get_xaxis().set_visible(False) for x in np.asarray(stim_axs2)[xaxes_off]]
[x.spines["bottom"].set_visible(False)
 for x in np.asarray(stim_axs1)[xaxes_off]]
[x.spines["bottom"].set_visible(False)
 for x in np.asarray(stim_axs2)[xaxes_off]]

# set all axes to same tick ranges
x_range = np.arange(0, 1.5, 0.5)
y_range = np.arange(-1, 2.1, 1)
[x.set_yticks(y_range) for x in np.asarray(stim_axs1)]
[x.set_xticks(x_range) for x in np.asarray(stim_axs1)]
[x.set_yticks(y_range) for x in np.asarray(stim_axs2)]
[x.set_xticks(x_range) for x in np.asarray(stim_axs2)]

# set bounds to make axes nicer
x_bounds = (0, 1)
y_bounds = (-1, 2)
[x.spines.left.set_bounds(y_bounds) for x in np.asarray(stim_axs1)]
[x.spines.bottom.set_bounds(x_bounds) for x in np.asarray(stim_axs1)]
[x.spines.left.set_bounds(y_bounds) for x in np.asarray(stim_axs2)]
[x.spines.bottom.set_bounds(x_bounds) for x in np.asarray(stim_axs2)]

# make axes shared
stim_axs1[0].get_shared_x_axes().join(stim_axs1[0], *stim_axs1)
stim_axs1[0].get_shared_y_axes().join(stim_axs1[0], *stim_axs1)
stim_axs2[0].get_shared_x_axes().join(stim_axs2[0], *stim_axs2)
stim_axs2[0].get_shared_y_axes().join(stim_axs2[0], *stim_axs2)

# add labels
subfig_l.supylabel('z-score')
subfig_l.supxlabel('green contrast')
subfig_r.supxlabel('red contrast')

# get legend handles
handles1, labels1 = stim_axs1[i].get_legend_handles_labels()
handles2, labels2 = stim_axs2[i].get_legend_handles_labels()

# add legends
subfig_l.legend(handles1, labels1, loc='upper center', ncol=2)
subfig_r.legend(handles2, labels2, loc='upper center', ncol=2)

# adjust margins
plt.subplots_adjust(left=0.175, right=0.975, top=0.87,
                    bottom=0.1, hspace=0.17, wspace=0.15)
fs.doublesave("../plots/contrast_curves")

plt.show()
