from copy import deepcopy

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
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

# compute the mean in each phase
mf.repeat_means(3)

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

# remove pauses
pause_index = np.arange(len(mf.ang_velocs))
