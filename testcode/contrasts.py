# load essential packages
import os
import matplotlib.pyplot as plt
import numpy as np
import modules.functions as fs
from dataclasses import dataclass
from vxtools.summarize.structure import SummaryFile
from sklearn.metrics import auc
from IPython import embed
from scipy.stats import pearsonr
from modules.dataloader import all_rois
from modules.plotstyle import PlotStyle
from modules.contrast import rg_activity, selective_rois

ps = PlotStyle()

# now load the data
f = SummaryFile('../data/Summary.hdf5')

# now select the recordings that were good
good_recs = [5, 8, 9, 10, 11, 13, 14]
# good_recs = [5, 14]

# load matrix of all rois of all layers with good rois
d = all_rois(f, good_recs)

datapath = "../data/"
dataname = "mean_dffs.npy"
timename = "mean_times.npy"

if os.path.exists(datapath+dataname) == False:
    d.stimulus_means()
    np.save(datapath+dataname, d.mean_dffs)
    np.save(datapath+timename, d.mean_times)
else:
    d.mean_dffs = np.load(datapath+dataname)
    d.mean_times = np.load(datapath+timename)

d.sort_means_by_corr()

# Threshold
counts, edges = np.histogram(d.corrs[:, 1], bins=50, density=True)

# compute a gaussian KDE
xkde, kde = fs.kde1d(d.corrs[:, 1], 0.02, xlims=[edges.min(), edges.max()])

target_auc = 0.2  # probability threshold

# create empty arrays for the gradient and index
gradient = np.zeros_like(xkde[:-1])
index = np.arange(len(xkde[:-1]))

# compute the gradient between the target and actual auc
for i in range(len(xkde[:-1])):
    area = auc(xkde[i:], kde[i:])
    gradient[i] = abs(area-target_auc)

# find the index where the gradient is smallest
idx = index[gradient == gradient.min()][0]

thresh = xkde[idx]

# get the rois
d.thresh_mean_rois = fs.thresh_correlations(d.corrs, thresh)

# compute mean across trials
d.repeat_means()

# get the phase and repeat mean dffs
d.active_meanmean_dffs = np.array(
    [d.meanstack_mean_dffs[int(roi), :] for roi in d.thresh_mean_rois[:, 0]])

# get the phase mean dff
d.active_mean_dffs = np.array([d.mean_dffs[int(roi), :]
                              for roi in d.thresh_mean_rois[:, 0]])

# get their rois as well
d.active_mean_rois = np.array([d.index_rois[int(roi)]
                              for roi in d.thresh_mean_rois[:, 0]])

# ... and the recording they came from
d.active_mean_recs = np.array([d.index_recs[int(roi)]
                              for roi in d.thresh_mean_rois[:, 0]])


# motion regressor
motion = np.array([1 if x != 0 else 0 for x in d.ang_velocs])

# clockwise motion regressor
clock = np.array([1 if x > 0 else 0 for x in d.ang_velocs])

# counterclockwise motion regressor
cclock = np.array([1 if x < 0 else 0 for x in d.ang_velocs])


corr_motion = np.array([pearsonr(x, motion)[0] for x in d.active_mean_dffs])

corr_clock = np.array([pearsonr(x, clock)[0] for x in d.active_mean_dffs])

corr_cclock = np.array([pearsonr(x, cclock)[0] for x in d.active_mean_dffs])


# make empty lists for contrasts
red_contr = []
green_contr = []
acr_contr = []

for rgb1, rgb2 in zip(d.rgb_1, d.rgb_2):
    red_contr.append(abs(rgb1[0]-rgb2[0]))
    green_contr.append(abs(rgb1[1]-rgb2[1]))

red_contr = np.array(red_contr)
green_contr = np.array(green_contr)

# iterate across channels and append chromatic contrasts
# for rgb1, rgb2 in zip(d.rgb_1, d.rgb_2):
#     if rgb2[1] == 0:
#         red_contr.append(rgb1[0])
#     else:
#         red_contr.append(0)
#     if rgb1[0] == 0:
#         green_contr.append(rgb2[1])
#     else:
#         green_contr.append(0)
# green_contr = np.array(green_contr)
# red_contr = np.array(red_contr)


# make achromatic contrast as the difference between the sums of both stripes
acr_contr = np.array([abs(np.sum(x)-np.sum(y))
                     for x, y in zip(d.rgb_1, d.rgb_2)])


corr_red = np.array([pearsonr(x, red_contr)[0] for x in d.active_mean_dffs])

corr_green = np.array([pearsonr(x, green_contr)[0]
                      for x in d.active_mean_dffs])

corr_acr = np.array([pearsonr(x, acr_contr)[0] for x in d.active_mean_dffs])


thresh = 0.3

# threshold rois

# get the data
dc = selective_rois(d, corr_clock, thresh)
dcc = selective_rois(d, corr_cclock, thresh)

# build stim
clock_stim = np.array(clock, dtype=float)
cclock_stim = np.array(cclock, dtype=float)

clock_stim[clock_stim == 0] = np.nan
cclock_stim[cclock_stim == 0] = np.nan

acr_clock_stim = clock_stim * acr_contr
acr_cclock_stim = cclock_stim * acr_contr


red_clock_stim = red_contr*clock_stim
green_clock_stim = green_contr*clock_stim

red_cclock_stim = red_contr*cclock_stim
green_cclock_stim = green_contr*cclock_stim

rg_clock_data = rg_activity(dc, red_clock_stim, green_clock_stim)
rg_cclock_data = rg_activity(dcc, red_cclock_stim, green_cclock_stim)

fig, ax = plt.subplots(2, 3, sharex=True, sharey=True,
                       figsize=(20*ps.cm, 20*ps.cm))
ax = fs.flatten(ax)

colors = [ps.orange, ps.red]
labels = ['clockwise-selective cells', 'countercl.-selective cells']
for lab, color, rg_clock in zip(labels, colors, [rg_clock_data, rg_cclock_data]):
    for i1, rds in enumerate(rg_clock.contr1):
        dffs = []
        contr = []
        for dff in np.array(rg_clock.dffs)[rg_clock.contr1 == rds]:
            for i in range(len(dff[:, 0])):

                # the data
                roi_dff = dff[i, :]
                contrsts2 = np.array(rg_clock.contr2_index)[
                    rg_clock.contr1 == rds][0]

                # sort the data
                sort_roi_dff = roi_dff[np.argsort(contrsts2)]
                sort_contrsts2 = contrsts2[np.argsort(contrsts2)]

                dffs.append(sort_roi_dff)
                contr.append(sort_contrsts2)

                ax[i1].scatter(
                    sort_contrsts2,
                    sort_roi_dff,
                    color='grey',
                    marker='.',
                    # edgecolor='gray',
                    alpha=0.1
                )

        # compute means for each category
        contrs = np.unique(fs.flatten(contr))
        meandff = []
        for c2 in contrs:
            meandff.append(np.mean(np.array(fs.flatten(dffs))
                           [fs.flatten(contr) == c2]))

        ax[i1].plot(contrs, meandff, label=lab, c=color, lw=2.5)
        ax[i1].set_title(f"Red contr = {np.round(rds,2)}", fontsize=14)

fig.supxlabel('Green channel contrast', fontsize=14)
fig.supylabel('Norm. dff', fontsize=14)
handles, labels = ax[i1].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', ncol=2)
plt.show()
