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


def dataconverter_std_ca(data, nresamples=10000, subsetsize_resampling=0.9):

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

            resamples = zscs
            # for i in range(nresamples):
            #     subset = np.random.choice(
            #         zscs, size=subsetsize, replace=False)
            #     resamples.append(subset)

            resamples = np.array(np.ravel(resamples))

            mean_zsc.append(np.mean(resamples))
            q025_zsc.append(np.mean(resamples) - np.std(resamples))
            q975_zsc.append(np.mean(resamples) + np.std(resamples))

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


def dataconverter_cis_ca(data, nresamples=10000, subsetsize_resampling=0.9):

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
                subset = np.random.choice(
                    zscs, size=subsetsize, replace=False)
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


def dataconverter_baseline_cis_ca(data, nresamples=10000, subsetsize_resampling=0.9):

    mean_zsc = []
    q025_zsc = []
    q975_zsc = []

    # get zscores for this condition
    zscs = np.ravel(data.zscores)

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


def dataconverter_eye(data, nresamples=10000, subsetsize_resampling=0.9):

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

            resamples = zscs
            # for i in range(nresamples):
            #     subset = np.random.choice(zscs, size=subsetsize, replace=False)
            #     resamples.append(subset)

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


def get_calcium():
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

    class combine_rg_activities:
        def __init__(self, rg_activities):

            self.zscores = np.concatenate(
                [x.zscores for x in rg_activities], axis=1)
            self.contr1_index = [x.contr1_index for x in rg_activities][0]
            self.contr2_index = [x.contr2_index for x in rg_activities][0]
            self.contr1 = [x.contr1 for x in rg_activities][0]
            self.contr2 = [x.contr2 for x in rg_activities][0]

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
        d2,
        d3
    ])

    # compute the mean in each phase
    mf.phase_means()

    # threshold the active ROIs

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

    # modify motion stim to remove all phases without motion
    clock = np.asarray([np.nan if x == 0 else x for x in clock])
    cclock = np.asarray([np.nan if x == 0 else x for x in cclock])

    # make regressors
    red_clock = clock * mf.red
    green_clock = clock * mf.green
    red_cclock = cclock * mf.red
    green_cclock = cclock * mf.green

    # extract activity at phases
    # rg_clock_data = rg_activity(mfclock, red_clock, green_clock)
    # rg_cclock_data = rg_activity(mfcclock, red_cclock, green_cclock)
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

    # make the bootstrap datasets
    rg_bootstr_clock = rg_activity(bootstr_data_clock, red_clock, green_clock)
    rg_bootstr_cclock = rg_activity(
        bootstr_data_cclock, red_cclock, green_cclock)
    gr_bootstr_clock = rg_activity(bootstr_data_clock, green_clock, red_clock)
    gr_bootstr_cclock = rg_activity(
        bootstr_data_cclock, green_cclock, red_cclock)

    # combine cells for both directions
    rg_bootstr = combine_rg_activities([rg_bootstr_cclock, rg_bootstr_clock])
    gr_bootstr = combine_rg_activities([gr_bootstr_cclock, gr_bootstr_clock])

    return gr_clock_data, gr_cclock_data, gr_bootstr


def get_eye():
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
    d2 = SingleFish(f2, good_recs2, overwrite=False, behav=True)
    d3 = SingleFish(f3, good_recs3, overwrite=False, behav=True)

    # test multifish class
    mf = MultiFish([d2, d3])

    for i in range(len(mf.eye_velocs[:, 0])):
        plt.plot(medfilt(np.ravel(mf.eye_velocs[i, :])+i, 21))
    plt.show()

    filt_velocs = []
    for rec in range(len(mf.eye_velocs[:, 0])):

        vels = mf.eye_velocs[rec, :]
        binlen = len(vels[0])
        vels_medilt = medfilt(np.ravel(vels), 41)

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

    return gr_eye


rg_clock_data, rg_cclock_data, rg_bootstr = get_calcium()
rg_eye = get_eye()

fig = plt.figure(figsize=(30*ps.cm, 14*ps.cm))

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
labels = ['clockw.-selective', 'countercl.-selective']

# get the data
contrast1, contrast2, mean_zscores_c, q025_zscores_c, q975_zscores_c = dataconverter_std_ca(
    rg_clock_data)
_, _, mean_zscores_cc, q025_zscores_cc, q975_zscores_cc = dataconverter_std_ca(
    rg_cclock_data)
# _, _, mean_zscores_brg, q025_zscores_brg, q975_zscores_brg = dataconverter_std(
#     rg_bootstr)

for i, ax in enumerate(stim_axs1):

    # plot mean lines
    ax.plot(contrast2[i], mean_zscores_c[i], lw=2, c=ps.red, label=labels[0])
    ax.plot(contrast2[i], mean_zscores_cc[i],
            lw=2, c=ps.orange, label=labels[1])
    # ax.plot(contrast2[i], mean_zscores_brg[i], lw=2, c=ps.blue)

    # plot bootstrapped confidence intervals
    ax.fill_between(contrast2[i], q025_zscores_c[i],
                    q975_zscores_c[i], alpha=0.3, color=ps.red, label='_')
    ax.fill_between(contrast2[i], q025_zscores_cc[i],
                    q975_zscores_cc[i], alpha=0.3, color=ps.orange, label='_')
    # ax.fill_between(contrast2[i], q025_zscores_brg[i],
    #                 q975_zscores_brg[i], alpha=0.2, color=ps.blue)

    # plot zero line
    ax.axhline(0, lw=1, ls='dashed', c='darkgray', label='_')
    ax.plot([np.round(rg_clock_data.contr1[i], 2), np.round(rg_clock_data.contr1[i], 2)], [-1, 2],
            lw=1, ls='dashed', label='_', color='darkgray', zorder=1000)
    ax.set_title(f"green: {np.round(rg_clock_data.contr1[i], 2)}", loc='left')

# plot the right side of the plot
colors = [ps.orange, ps.red, ps.blue]
labels = ['clockw.-selective', 'countercl.-selective', 'bootstrap']

# get the data
contrast1, contrast2, mean_zscores_c, q025_zscores_c, q975_zscores_c = dataconverter_eye(
    rg_eye)
# _, _, mean_zscores_cc, q025_zscores_cc, q975_zscores_cc = dataconverter_std(
#     gr_cclock_data)
# _, _, mean_zscores_bgr, q025_zscores_bgr, q975_zscores_bgr = dataconverter_std(
#     rg_bootstr)

for i, ax in enumerate(stim_axs2):

    # plot mean lines
    ax.plot(contrast2[i], mean_zscores_c[i],
            color=ps.c1, lw=2, label='behavior')
    #ax.plot(contrast2[i], mean_zscores_cc[i], color=ps.c2, lw=2)
    #ax.plot(contrast2[i], mean_zscores_bgr[i])

    # plot bootstrapped confidence intervals
    ax.fill_between(contrast2[i], q025_zscores_c[i],
                    q975_zscores_c[i], alpha=0.3, color=ps.c1, label='_')
    # ax.fill_between(contrast2[i], q025_zscores_cc[i],
    #                 q975_zscores_cc[i], alpha=0.2, color=ps.c2)
    # ax.fill_between(contrast2[i], q025_zscores_bgr[i],
    #                 q975_zscores_bgr[i], alpha=0.2)

    # plot zero line
    # ax.axhline(0, lw=1, ls='dashed', c='darkgray')
    ax.set_title(f"green: {np.round(rg_clock_data.contr1[i], 2)}", loc='left')
    ax.plot([np.round(rg_clock_data.contr1[i], 2), np.round(rg_clock_data.contr1[i], 2)], [0, 0.2],
            lw=1, ls='dashed', label='_', color='darkgray', zorder=1000)
    ax.axhline(0, lw=1, ls='dashed', c='darkgray', label='_')

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
[x.set_yticks(np.arange(0, 0.21, 0.05)) for x in np.asarray(stim_axs2)]
[x.set_xticks(x_range) for x in np.asarray(stim_axs2)]

# set bounds to make axes nicer
x_bounds = (0, 1)
y_bounds = (-1, 2)
[x.spines.left.set_bounds(y_bounds) for x in np.asarray(stim_axs1)]
[x.spines.bottom.set_bounds(x_bounds) for x in np.asarray(stim_axs1)]
[x.spines.left.set_bounds((0, 0.2)) for x in np.asarray(stim_axs2)]
[x.spines.bottom.set_bounds(x_bounds) for x in np.asarray(stim_axs2)]

# make axes shared
stim_axs1[0].get_shared_x_axes().join(stim_axs1[0], *stim_axs1)
stim_axs1[0].get_shared_y_axes().join(stim_axs1[0], *stim_axs1)
stim_axs2[0].get_shared_x_axes().join(stim_axs2[0], *stim_axs2)
stim_axs2[0].get_shared_y_axes().join(stim_axs2[0], *stim_axs2)

# add labels
subfig_l.supylabel('mean z-score', fontsize=14)
subfig_r.supylabel('mean eye velocity [°/s]', fontsize=14)
subfig_r.supxlabel('red contrast', fontsize=14)
subfig_l.supxlabel('red contrast', fontsize=14)
# subfig_l.suptitle('Calcium activity')
# subfig_r.suptitle('Behavior')

# get legend handles
handles1, labels1 = stim_axs1[i].get_legend_handles_labels()
handles2, labels2 = stim_axs2[i].get_legend_handles_labels()
handles1.extend(handles2)
labels1.extend(labels2)
# add legends

subfig_r.legend(handles1, labels1, loc=(-0.6, 0.94), ncol=3)

# adjust margins
plt.subplots_adjust(left=0.175, right=0.930, top=0.850,
                    bottom=0.125, hspace=0.200, wspace=0.050)
fs.doublesave("../plots/contrast_curves")
plt.show()
