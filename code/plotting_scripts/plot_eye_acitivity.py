from copy import deepcopy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from IPython import embed
from scipy.signal import medfilt
from scipy.stats import pearsonr
from sklearn.metrics import auc
from tqdm import tqdm
from vxtools.summarize.structure import SummaryFile

import modules.functions as fs
from modules.contrast import (phase_activity, rg_activity_trash,
                              selective_rois_trash)
from modules.dataloader import MultiFish, SingleFish
from modules.plotstyle import PlotStyle

ps = PlotStyle()


def dataconverter(data, nresamples=10000, subsetsize_resampling=0.9):

    contrast1 = []
    contrast2 = []
    mean_eye_v = []
    q025_zscores = []
    q975_zscores = []

    # iterate through contrast levels
    for it1, c1 in tqdm(enumerate(data.contr1)):

        # collect zscore and contrast data here
        zscores = []
        contrasts1 = []
        contrasts2 = []

        # for each contrast1 level, go through all contrast2 levels
        for zsc in np.array(data.eye_velocs)[data.contr1 == c1]:

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
            q025_zsc.append(np.mean(resamples) - np.std(resamples))
            q975_zsc.append(np.mean(resamples) + np.std(resamples))

        mean_zsc = np.asarray(mean_zsc)
        q025_zsc = np.asarray(q025_zsc)
        q975_zsc = np.asarray(q975_zsc)

        mean_eye_v.append(mean_zsc)
        q025_zscores.append(q025_zsc)
        q975_zscores.append(q975_zsc)

        contrast1.append(np.ones_like(contr2_lvls)*c1)
        contrast2.append(contr2_lvls)

    contrast1 = np.asarray(contrast1)
    contrast2 = np.asarray(contrast2)
    mean_eye_v = np.asarray(mean_eye_v)
    q025_zscores = np.asarray(q025_zscores)
    q975_zscores = np.asarray(q975_zscores)

    return contrast1, contrast2, mean_eye_v, q025_zscores, q975_zscores


class rg_activity:
    def __init__(self, selective_rois, contr1, contr2):

        self.__rois = selective_rois
        self.__contr1 = contr1
        self.__contr2 = contr2
        self.__index = np.arange(len(self.__contr1))
        self.contr1 = np.unique(self.__contr1[~np.isnan(self.__contr1)])
        self.contr2 = np.unique(self.__contr2[~np.isnan(self.__contr2)])
        self.eye_velocs = []
        # self.mean_dffs = []
        self.contr1_index = []
        self.contr2_index = []
        self.rois = self.__rois.rois
        self.recs = self.__rois.recs

        for c1 in self.contr1:

            self.contr1_index.append(c1)
            idx = self.__index[self.__contr1 == c1]
            cat_zscores = self.__rois.eye_velocs[:, idx]
            # mean_dffs = np.mean(cat_dffs, axis=1)
            self.contr2_index.append(self.__contr2[idx])
            self.eye_velocs.append(cat_zscores)
            # self.mean_dffs.append(mean_dffs)

        # self.mean_dffs = np.array(self.mean_dffs)
        self.contr1_index = np.array(self.contr1_index)
        self.contr2_index = np.array(self.contr2_index)


# now load the data
DATAROOT2 = '../data/data2/'
DATAROOT3 = '../data/data3/'

f2 = SummaryFile(DATAROOT2 + 'Summary.hdf5')
f3 = SummaryFile(DATAROOT3 + 'Summary.hdf5')

good_recs2 = [0, 1, 2, 4, 5]
good_recs3 = [0, 1, 2, 3, 4]

# load matrix of all rois of all layers with good rois
d2 = SingleFish(f2, good_recs2, overwrite=True, behav=True)
d3 = SingleFish(f3, good_recs3, overwrite=True, behav=True)

# test multifish class
mf = MultiFish([d2, d3])

for i in range(len(mf.eye_velocs[:, 0])):
    plt.plot(medfilt(np.ravel(mf.eye_velocs[i, :])+i, 21))
plt.show()

filt_velocs = []
for rec in range(len(mf.eye_velocs[:, 0])):

    vels = mf.eye_velocs[rec, :]
    binlen = len(vels[0])
    vels_medilt = medfilt(np.ravel(vels), 45)

    totlen = len(vels_medilt)
    numwind = totlen / binlen

    steps = np.arange(0, totlen, binlen)

    new_mvels = [np.asarray(vels_medilt[np.arange(x, x+binlen)])
                 for x in steps]
    filt_velocs.append(np.asarray(new_mvels))

mf.eye_velocs = np.array(filt_velocs)

# test phase mean computation
mf.phase_means()

# make motion regressor
motion = np.array([1 if x != 0 else np.nan for x in mf.ang_velocs])

# combine with red and green
red_motion = motion * mf.red
green_motion = motion * mf.green

# cluster data by phase information
rg_eye = rg_activity(mf, red_motion, green_motion)
gr_eye = rg_activity(mf, green_motion, red_motion)

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
contrast1, contrast2, mean_v, q025_v, q975_v = dataconverter(rg_eye)

for i, ax in enumerate(stim_axs1):

    # plot mean lines
    ax.plot(contrast2[i], mean_v[i], lw=2, c=ps.red)

    # plot bootstrapped confidence intervals
    ax.fill_between(contrast2[i], q025_v[i],
                    q975_v[i], alpha=0.2, color=ps.red)

    # plot zero line
    ax.axhline(0, lw=1, ls='dashed', c='darkgray')

# plot the right side of the plot
colors = [ps.orange, ps.red, ps.blue]
labels = ['clockw.-selective', 'countercl.-selective', 'bootstrap']

# get the data
contrast1, contrast2, mean_v, q025_v, q975_v = dataconverter(gr_eye)

for i, ax in enumerate(stim_axs2):

    # plot mean lines
    ax.plot(contrast2[i], mean_v[i], color=ps.c1, lw=2)
    # ax.plot(contrast2[i], mean_zscores_bgr[i])

    # plot bootstrapped confidence intervals
    ax.fill_between(contrast2[i], q025_v[i],
                    q975_v[i], alpha=0.2, color=ps.c1)

    # plot zero line
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

# # set all axes to same tick ranges
# x_range = np.arange(0, 1.5, 0.5)
# y_range = np.arange(-1, 3.1, 1)
# [x.set_yticks(y_range) for x in np.asarray(stim_axs1)]
# [x.set_xticks(x_range) for x in np.asarray(stim_axs1)]
# [x.set_yticks(y_range) for x in np.asarray(stim_axs2)]
# [x.set_xticks(x_range) for x in np.asarray(stim_axs2)]

# # set bounds to make axes nicer
# x_bounds = (0, 1)
# y_bounds = (-1, 3)
# [x.spines.left.set_bounds(y_bounds) for x in np.asarray(stim_axs1)]
# [x.spines.bottom.set_bounds(x_bounds) for x in np.asarray(stim_axs1)]
# [x.spines.left.set_bounds(y_bounds) for x in np.asarray(stim_axs2)]
# [x.spines.bottom.set_bounds(x_bounds) for x in np.asarray(stim_axs2)]

# add labels
subfig_l.supylabel('z-score')
subfig_l.supxlabel('red contrast')
subfig_r.supxlabel('green contrast')

# get legend handles
handles1, labels1 = stim_axs1[i].get_legend_handles_labels()
handles2, labels2 = stim_axs2[i].get_legend_handles_labels()

# add legends
subfig_l.legend(handles1, labels1, loc='upper center', ncol=2)
subfig_r.legend(handles2, labels2, loc='upper center', ncol=2)

# adjust margins
plt.subplots_adjust(left=0.175, right=0.975, top=0.87,
                    bottom=0.1, hspace=0.17, wspace=0.15)
fs.doublesave("../plots/contrast_curves_behav")
plt.show()
